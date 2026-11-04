"""MAPPO-only mixed resistive/CPL training model.

This module deliberately lives under ``MAPPO训练`` so the established
RMAPPO and resistive-load training environment remains unchanged.  Every bus
keeps its configured resistive base load.  During each episode, one randomly
selected bus carries a baseline constant-power load, steps to a target power,
and later returns to the baseline.  The Actor observation and action contracts
are unchanged.
"""

from __future__ import annotations

import numpy as np

from onpolicy.envs.dc_microgrid.secondary_doubleloop_model import (
    DCMicrogridModel,
)


class CPLDCMicrogridModel(DCMicrogridModel):
    """Four-bus averaged model with an event-driven CPL power step."""

    def __init__(self, args):
        # Keep the Actor interface unchanged (six observations and one action
        # per agent), but assign the two secondary-control jobs explicitly:
        # the zero-mean Actor action controls current sharing, while this
        # common PI restores the average bus voltage.
        self.common_voltage_kp = float(
            getattr(args, "dc_common_voltage_kp", 0.5)
        )
        self.common_voltage_ki = float(
            getattr(args, "dc_common_voltage_ki", 2.0)
        )
        self.common_voltage_min_v = float(
            getattr(args, "dc_common_voltage_min_v", -1.0)
        )
        self.common_voltage_max_v = float(
            getattr(args, "dc_common_voltage_max_v", 3.0)
        )
        self.common_action_filter_alpha = float(
            getattr(args, "dc_common_action_filter_alpha", 0.2)
        )
        self.w_voltage_violation = float(
            getattr(args, "dc_reward_voltage_violation", 20.0)
        )
        self.w_current_violation = float(
            getattr(args, "dc_reward_current_violation", 3.0)
        )
        if self.common_voltage_kp < 0.0 or self.common_voltage_ki < 0.0:
            raise ValueError("common-voltage PI gains must be non-negative")
        if self.common_voltage_min_v >= self.common_voltage_max_v:
            raise ValueError(
                "dc_common_voltage_min_v must be below dc_common_voltage_max_v"
            )
        if not 0.0 < self.common_action_filter_alpha <= 1.0:
            raise ValueError("dc_common_action_filter_alpha must be in (0, 1]")
        if self.w_voltage_violation < 0.0 or self.w_current_violation < 0.0:
            raise ValueError("separate violation weights must be non-negative")
        self.common_voltage_integral_v = 0.0
        self.common_voltage_action_v = 0.0
        self.common_voltage_filtered_v = 0.0
        self.actor_action_mean_v = 0.0
        self.actor_differential_actions = np.zeros(4, dtype=np.float64)
        self.actor_differential_filtered_actions = np.zeros(4, dtype=np.float64)

        # Keep the default at zero so historical 0 -> CPL -> 0 checkpoints are
        # evaluated with their original physical scenario.  The new fine-tune
        # configuration explicitly selects a 100 W baseline.
        self.cpl_base_power_w = float(
            getattr(args, "dc_cpl_base_power_w", 0.0)
        )
        self.cpl_power_min_w = float(getattr(args, "dc_cpl_power_min_w", 100.0))
        self.cpl_power_max_w = float(getattr(args, "dc_cpl_power_max_w", 300.0))
        self.cpl_voltage_floor_v = float(
            getattr(args, "dc_cpl_voltage_floor_v", 20.0)
        )
        self.cpl_solver_max_iterations = int(
            getattr(args, "dc_cpl_solver_max_iterations", 12)
        )
        self.cpl_solver_tolerance_a = float(
            getattr(args, "dc_cpl_solver_tolerance_a", 1.0e-7)
        )
        self.cpl_low_voltage_v = float(
            getattr(args, "dc_cpl_low_voltage_v", 43.2)
        )
        self.cpl_current_limit_a = float(
            getattr(args, "dc_cpl_current_limit_a", 10.0)
        )
        self.w_cpl_low_voltage = float(
            getattr(args, "dc_reward_cpl_low_voltage", 5.0)
        )
        self.w_cpl_overcurrent = float(
            getattr(args, "dc_reward_cpl_overcurrent", 2.0)
        )
        self.w_cpl_solver_failure = float(
            getattr(args, "dc_reward_cpl_solver_failure", 50.0)
        )

        if self.cpl_base_power_w < 0.0:
            raise ValueError("dc_cpl_base_power_w must be non-negative")
        if self.cpl_power_min_w <= 0.0:
            raise ValueError("dc_cpl_power_min_w must be positive")
        if self.cpl_power_max_w < self.cpl_power_min_w:
            raise ValueError("dc_cpl_power_max_w must be >= dc_cpl_power_min_w")
        if self.cpl_base_power_w > self.cpl_power_min_w:
            raise ValueError(
                "dc_cpl_base_power_w must not exceed dc_cpl_power_min_w"
            )
        if self.cpl_voltage_floor_v <= 0.0:
            raise ValueError("dc_cpl_voltage_floor_v must be positive")
        if self.cpl_solver_max_iterations < 1:
            raise ValueError("dc_cpl_solver_max_iterations must be positive")
        if self.cpl_solver_tolerance_a <= 0.0:
            raise ValueError("dc_cpl_solver_tolerance_a must be positive")
        if self.cpl_low_voltage_v <= self.cpl_voltage_floor_v:
            raise ValueError("dc_cpl_low_voltage_v must exceed the numerical floor")
        if self.cpl_current_limit_a <= 0.0:
            raise ValueError("dc_cpl_current_limit_a must be positive")
        if min(
            self.w_cpl_low_voltage,
            self.w_cpl_overcurrent,
            self.w_cpl_solver_failure,
        ) < 0.0:
            raise ValueError("CPL reward weights must be non-negative")

        self.cpl_powers_w = np.zeros(4, dtype=np.float64)
        self.pre_step_cpl_powers_w = None
        self.active_cpl_power_w = self.cpl_power_min_w
        self.cpl_solver_converged = True
        self.cpl_interval_solver_converged = True
        self.cpl_solver_iterations = 0
        self.cpl_interval_max_iterations = 0
        super().__init__(args)

    def _refresh_bus_solver(self):
        """Build the linear network part; CPL current is handled by Newton."""
        conductance = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            conductance[i, i] = (
                1.0 / self.source_resistances[i]
                + 1.0 / max(self.load_resistances[i], 1.0e-6)
                + np.sum(self.adjacency[i]) / self.line_resistance
            )
            for j in np.where(self.adjacency[i] > 0.0)[0]:
                conductance[i, j] = -1.0 / self.line_resistance
        self._bus_conductance = conductance
        # Retain the legacy member for diagnostics and the zero-CPL fast path.
        self._bus_inverse = np.linalg.inv(conductance)

    def _solve_bus_voltages(self):
        """Solve G*V + P/V = Vs/Rs using damped Newton iterations."""
        rhs = self.converter_voltages / self.source_resistances
        if not np.any(self.cpl_powers_w > 0.0):
            voltage = np.matmul(self._bus_inverse, rhs)
            self._record_cpl_solver_status(True, 0)
            return voltage

        linear_voltage = np.matmul(self._bus_inverse, rhs)
        previous = getattr(self, "bus_voltages", linear_voltage)
        if np.all(np.isfinite(previous)) and np.min(previous) > self.cpl_voltage_floor_v:
            voltage = np.asarray(previous, dtype=np.float64).copy()
        else:
            voltage = np.asarray(linear_voltage, dtype=np.float64).copy()
        voltage = np.maximum(voltage, self.cpl_voltage_floor_v)

        converged = False
        iterations = 0
        for iterations in range(1, self.cpl_solver_max_iterations + 1):
            safe_voltage = np.maximum(voltage, self.cpl_voltage_floor_v)
            residual = (
                np.matmul(self._bus_conductance, voltage)
                + self.cpl_powers_w / safe_voltage
                - rhs
            )
            residual_norm = float(np.max(np.abs(residual)))
            if residual_norm <= self.cpl_solver_tolerance_a:
                converged = True
                break

            jacobian = self._bus_conductance - np.diag(
                self.cpl_powers_w / (safe_voltage ** 2)
            )
            try:
                delta = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                break

            accepted = False
            damping = 1.0
            for _ in range(12):
                trial = voltage + damping * delta
                if np.min(trial) <= self.cpl_voltage_floor_v:
                    damping *= 0.5
                    continue
                trial_residual = (
                    np.matmul(self._bus_conductance, trial)
                    + self.cpl_powers_w / trial
                    - rhs
                )
                if float(np.max(np.abs(trial_residual))) < residual_norm:
                    voltage = trial
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                break

        if not converged and np.all(np.isfinite(voltage)):
            safe_voltage = np.maximum(voltage, self.cpl_voltage_floor_v)
            final_residual = (
                np.matmul(self._bus_conductance, voltage)
                + self.cpl_powers_w / safe_voltage
                - rhs
            )
            converged = bool(
                np.max(np.abs(final_residual))
                <= 10.0 * self.cpl_solver_tolerance_a
            )

        self._record_cpl_solver_status(converged, iterations)
        return voltage

    def _record_cpl_solver_status(self, converged, iterations):
        self.cpl_solver_converged = bool(converged)
        self.cpl_solver_iterations = int(iterations)
        self.cpl_interval_solver_converged = bool(
            getattr(self, "cpl_interval_solver_converged", True) and converged
        )
        self.cpl_interval_max_iterations = max(
            int(getattr(self, "cpl_interval_max_iterations", 0)),
            int(iterations),
        )

    def _reset_interval_voltage_extrema(self):
        super()._reset_interval_voltage_extrema()
        self.cpl_interval_solver_converged = True
        self.cpl_interval_max_iterations = 0

    def reset(self):
        self.cpl_powers_w = np.zeros(4, dtype=np.float64)
        self.pre_step_cpl_powers_w = None
        if np.isclose(self.cpl_power_min_w, self.cpl_power_max_w):
            self.active_cpl_power_w = self.cpl_power_min_w
        else:
            self.active_cpl_power_w = float(
                self.rng.uniform(self.cpl_power_min_w, self.cpl_power_max_w)
            )
        # Draw the episode's stepped CPL power before reset() produces the
        # noisy first observation.  This keeps normal and noise0 evaluations
        # on exactly the same physical load for a given seed.
        super().reset()

        # The parent reset chooses the event bus near its end.  Install the
        # baseline CPL on that same bus, then rebuild all electrical states so
        # the first returned observation is physically consistent with the
        # requested baseline rather than with a zero-CPL initialization.
        self.cpl_powers_w[self.active_load_step_agent] = self.cpl_base_power_w
        self.bus_voltages = self._solve_bus_voltages()
        self.output_currents = (
            self.converter_voltages - self.bus_voltages
        ) / self.source_resistances
        self.inductor_currents = np.maximum(self.output_currents, 0.0)
        self.load_currents = self._mixed_load_currents(self.bus_voltages)
        self.previous_per_capacity_currents = (
            self.inductor_currents / self.capacity_ratios
        ).copy()
        self.current_references = self.inductor_currents.copy()
        self.voltage_pi_integral = self.current_references.copy()
        # Bumpless start: compensate the primary droop already implied by the
        # pre-energised branch currents instead of ramping the common action
        # from zero during the first seconds of every episode.
        initial_common_v = float(
            np.clip(
                np.mean(self.droop_coefficients * self.inductor_currents),
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )
        self.common_voltage_integral_v = initial_common_v
        self.common_voltage_action_v = initial_common_v
        self.common_voltage_filtered_v = initial_common_v
        self.actor_action_mean_v = 0.0
        self.actor_differential_actions = np.zeros(4, dtype=np.float64)
        self.actor_differential_filtered_actions = np.zeros(4, dtype=np.float64)
        self.secondary_actions = np.full(4, initial_common_v, dtype=np.float64)
        self.raw_secondary_actions = self.secondary_actions.copy()
        self.previous_secondary_actions = self.secondary_actions.copy()
        self._reset_interval_voltage_extrema()
        return self._get_obs()

    def _apply_secondary_action(self, raw_actions):
        """Combine zero-mean MAPPO sharing action with average-voltage PI."""
        actor_raw = np.asarray(raw_actions, dtype=np.float64).reshape(4, -1)[:, 0]
        self.actor_action_mean_v = float(np.mean(actor_raw))
        self.actor_differential_actions = actor_raw - self.actor_action_mean_v

        average_voltage_v = float(np.mean(self.bus_voltages))
        voltage_error_v = self.nominal_voltage - average_voltage_v
        self.common_voltage_integral_v = float(
            np.clip(
                self.common_voltage_integral_v
                + self.common_voltage_ki * voltage_error_v * self.secondary_dt,
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )
        self.common_voltage_action_v = float(
            np.clip(
                self.common_voltage_kp * voltage_error_v
                + self.common_voltage_integral_v,
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )

        self.actor_differential_filtered_actions += self.action_filter_alpha * (
            self.actor_differential_actions
            - self.actor_differential_filtered_actions
        )
        self.common_voltage_filtered_v += self.common_action_filter_alpha * (
            self.common_voltage_action_v - self.common_voltage_filtered_v
        )
        combined_target = (
            self.common_voltage_action_v + self.actor_differential_actions
        )
        combined_applied = (
            self.common_voltage_filtered_v
            + self.actor_differential_filtered_actions
        )
        self.raw_secondary_actions = combined_target.copy()
        self.previous_secondary_actions = self.secondary_actions.copy()
        self.secondary_actions = combined_applied.copy()
        self.last_action_change = float(
            np.mean(np.abs(self.secondary_actions - self.previous_secondary_actions))
        )

    def _update_load(self):
        """Step the CPL to its target and later restore its baseline power."""
        self.load_event_this_step = False
        if self.load_random_walk > 0.0:
            self.load_resistances += self.rng.normal(0.0, self.load_random_walk, 4)
            self.load_resistances = np.clip(
                self.load_resistances,
                self.load_resistance_min,
                self.load_resistance_max,
            )

        if self.t == self.active_load_step_time:
            self.pre_step_cpl_powers_w = self.cpl_powers_w.copy()
            self.cpl_powers_w[self.active_load_step_agent] = self.active_cpl_power_w
            self.last_event_step = self.t
            self.last_transient_event_step = self.t
            self.load_event_this_step = True
        if (
            self.t == self.active_load_restore_time
            and self.pre_step_cpl_powers_w is not None
        ):
            self.cpl_powers_w = self.pre_step_cpl_powers_w.copy()
            self.pre_step_cpl_powers_w = None
            self.last_event_step = self.t
            self.last_transient_event_step = self.t
            self.load_event_this_step = True
        self._refresh_bus_solver()

    def _mixed_load_currents(self, voltages):
        safe_voltage = np.maximum(
            np.asarray(voltages, dtype=np.float64),
            self.cpl_voltage_floor_v,
        )
        return (
            np.asarray(voltages, dtype=np.float64) / self.load_resistances
            + self.cpl_powers_w / safe_voltage
        )

    def _plant_step(self, record_pre_update=False):
        super()._plant_step(record_pre_update=record_pre_update)
        self.load_currents = self._mixed_load_currents(self.bus_voltages)

    def _reward(self):
        _, info = super()._reward()
        average_voltage_error_v = abs(
            float(np.mean(self.bus_voltages)) - self.nominal_voltage
        )
        average_voltage_error = average_voltage_error_v / self.nominal_voltage
        current_error = float(info["current_error"])
        voltage_violation = max(
            0.0,
            (average_voltage_error_v - self.average_voltage_band_v)
            / self.average_voltage_band_v,
        )
        current_violation = max(
            0.0,
            (current_error - self.current_target_error)
            / max(self.current_target_error, 1.0e-6),
        )
        weighted_constraint_penalty = (
            self.w_voltage_violation * voltage_violation
            + self.w_current_violation * current_violation
        )
        reward = (
            -self.w_average_voltage * average_voltage_error
            -self.w_current_share * current_error
            -weighted_constraint_penalty
            -float(info["weighted_transient_voltage_penalty"])
            -float(info["weighted_steady_stability_penalty"])
        )
        minimum_voltage_v = float(np.min(self.interval_bus_voltage_min_v))
        maximum_current_a = float(np.max(np.abs(self.inductor_currents)))
        low_voltage_excess = max(
            0.0,
            (self.cpl_low_voltage_v - minimum_voltage_v)
            / self.cpl_low_voltage_v,
        )
        overcurrent_excess = max(
            0.0,
            (maximum_current_a - self.cpl_current_limit_a)
            / self.cpl_current_limit_a,
        )
        solver_failure = float(not self.cpl_interval_solver_converged)
        cpl_safety_penalty = (
            self.w_cpl_low_voltage * low_voltage_excess ** 2
            + self.w_cpl_overcurrent * overcurrent_excess ** 2
            + self.w_cpl_solver_failure * solver_failure
        )
        reward -= cpl_safety_penalty
        info.update(
            {
                "load_model": "resistive_plus_cpl",
                "cpl_base_power_w": float(self.cpl_base_power_w),
                "active_cpl_power_w": float(self.active_cpl_power_w),
                "total_cpl_power_w": float(np.sum(self.cpl_powers_w)),
                "cpl_step_agent": int(self.active_load_step_agent),
                "cpl_minimum_bus_voltage_v": minimum_voltage_v,
                "cpl_maximum_inductor_current_a": maximum_current_a,
                "cpl_solver_converged": float(
                    self.cpl_interval_solver_converged
                ),
                "cpl_solver_max_iterations": int(
                    self.cpl_interval_max_iterations
                ),
                "cpl_low_voltage_penalty": float(low_voltage_excess ** 2),
                "cpl_overcurrent_penalty": float(overcurrent_excess ** 2),
                "weighted_cpl_safety_penalty": float(cpl_safety_penalty),
                "common_voltage_error_v": float(
                    self.nominal_voltage - np.mean(self.bus_voltages)
                ),
                "common_voltage_integral_v": float(
                    self.common_voltage_integral_v
                ),
                "common_voltage_action_v": float(self.common_voltage_action_v),
                "common_voltage_filtered_v": float(
                    self.common_voltage_filtered_v
                ),
                "common_action_filter_alpha": float(
                    self.common_action_filter_alpha
                ),
                "actor_action_mean_v": float(self.actor_action_mean_v),
                "voltage_violation": float(voltage_violation),
                "current_violation": float(current_violation),
                "weighted_constraint_penalty": float(
                    weighted_constraint_penalty
                ),
            }
        )
        for agent_id in range(self.num_agents):
            info[f"cpl_power_{agent_id}_w"] = float(
                self.cpl_powers_w[agent_id]
            )
            info[f"actor_differential_action_{agent_id}_v"] = float(
                self.actor_differential_actions[agent_id]
            )
            info[f"actor_differential_filtered_{agent_id}_v"] = float(
                self.actor_differential_filtered_actions[agent_id]
            )
        return float(reward), info

    def step(self, raw_actions):
        observation, reward, done, info = super().step(raw_actions)
        if not self.cpl_interval_solver_converged:
            reward = -100.0
            done = True
        return observation, reward, done, info

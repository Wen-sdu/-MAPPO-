"""Unified resistive/CPL environment for feed-forward MAPPO training.

The Actor contract intentionally remains unchanged: every agent observes the
same six locally available current/voltage features and emits one secondary
voltage action.  The Actor mean is removed so MAPPO concentrates on current
sharing, while a cheap common-voltage PI is evaluated at the 1 kHz voltage
loop rate.  This keeps the learned network compatible with the existing DSP
exporter and keeps neural-network inference at 100 Hz.
"""

from __future__ import annotations

import math

import numpy as np

from cpl_model import CPLDCMicrogridModel
from onpolicy.envs.dc_microgrid.secondary_doubleloop_model import DCMicrogridModel


class MixedLoadDCMicrogridModel(CPLDCMicrogridModel):
    """Four-bus model with random resistive, CPL, and mixed load events."""

    def __init__(self, args):
        # Scenario envelope.  These attributes must exist before the parent
        # constructor calls the virtual reset() method.
        self.mixed_resistance_initial_min = float(
            getattr(args, "dc_mixed_resistance_initial_min_ohm", 40.0)
        )
        self.mixed_resistance_initial_max = float(
            getattr(args, "dc_mixed_resistance_initial_max_ohm", 80.0)
        )
        self.mixed_resistance_target_min = float(
            getattr(args, "dc_mixed_resistance_target_min_ohm", 20.0)
        )
        self.mixed_resistance_target_max = float(
            getattr(args, "dc_mixed_resistance_target_max_ohm", 80.0)
        )
        self.mixed_cpl_initial_max_w = float(
            getattr(args, "dc_mixed_cpl_initial_max_w", 100.0)
        )
        self.mixed_cpl_target_max_w = float(
            getattr(args, "dc_mixed_cpl_target_max_w", 500.0)
        )
        self.mixed_total_load_power_max_w = float(
            getattr(args, "dc_mixed_total_load_power_max_w", 900.0)
        )
        self.mixed_event_bus_count_min = int(
            getattr(args, "dc_mixed_event_bus_count_min", 1)
        )
        self.mixed_event_bus_count_max = int(
            getattr(args, "dc_mixed_event_bus_count_max", 4)
        )
        self.mixed_resistive_probability = float(
            getattr(args, "dc_mixed_resistive_probability", 0.30)
        )
        self.mixed_cpl_probability = float(
            getattr(args, "dc_mixed_cpl_probability", 0.35)
        )
        self.mixed_combined_probability = float(
            getattr(args, "dc_mixed_combined_probability", 0.35)
        )
        self.mixed_ramp_probability = float(
            getattr(args, "dc_mixed_ramp_probability", 0.35)
        )
        self.mixed_ramp_steps_min = int(
            getattr(args, "dc_mixed_ramp_steps_min", 2)
        )
        self.mixed_ramp_steps_max = int(
            getattr(args, "dc_mixed_ramp_steps_max", 10)
        )

        num_env_steps = max(1, int(getattr(args, "num_env_steps", 500000)))
        rollout_threads = max(1, int(getattr(args, "n_rollout_threads", 1)))
        episode_length = max(1, int(getattr(args, "episode_length", 2000)))
        default_curriculum_episodes = int(
            math.ceil(num_env_steps / float(rollout_threads * episode_length))
        )
        requested_curriculum_episodes = int(
            getattr(args, "dc_mixed_curriculum_episodes", 0)
        )
        self.mixed_curriculum_episodes = max(
            1,
            requested_curriculum_episodes or default_curriculum_episodes,
        )
        self._mixed_reset_count = -1

        # The common path is deliberately deterministic and uses only Vdc,
        # which is available in HIL.  It is much faster than Actor inference.
        self.fast_common_voltage_enabled = bool(
            getattr(args, "dc_fast_common_voltage", False)
        )
        self.fast_common_action_filter_alpha = float(
            getattr(args, "dc_fast_common_action_filter_alpha", 0.10)
        )
        self.common_current_feedforward_gain = float(
            getattr(args, "dc_common_current_feedforward_gain", 0.0)
        )
        self.common_feedforward_filter_alpha = float(
            getattr(args, "dc_common_feedforward_filter_alpha", 0.5)
        )

        # Extra safety/transient objectives for the wider operating envelope.
        self.mixed_average_nadir_v = float(
            getattr(args, "dc_mixed_average_nadir_v", 43.2)
        )
        self.mixed_average_zenith_v = float(
            getattr(args, "dc_mixed_average_zenith_v", 52.8)
        )
        self.w_mixed_average_nadir = float(
            getattr(args, "dc_reward_mixed_average_nadir", 20.0)
        )
        self.w_mixed_average_zenith = float(
            getattr(args, "dc_reward_mixed_average_zenith", 20.0)
        )
        self.w_mixed_transient_iae = float(
            getattr(args, "dc_reward_mixed_transient_iae", 8.0)
        )
        self.w_mixed_duty_saturation = float(
            getattr(args, "dc_reward_mixed_duty_saturation", 4.0)
        )
        self.w_mixed_common_saturation = float(
            getattr(args, "dc_reward_mixed_common_saturation", 1.0)
        )

        self.initial_load_resistances = np.full(4, 40.0, dtype=np.float64)
        self.target_load_resistances = self.initial_load_resistances.copy()
        self.initial_cpl_powers_w = np.zeros(4, dtype=np.float64)
        self.target_cpl_powers_w = np.zeros(4, dtype=np.float64)
        self.mixed_event_bus_mask = np.zeros(4, dtype=bool)
        self.mixed_scenario_kind = "resistive"
        self.mixed_curriculum_progress = 0.0
        self.active_load_ramp_steps = 1
        self.interval_average_abs_error_sum_v = 0.0
        self.interval_voltage_sample_count = 0
        self.common_saturation_samples = 0
        self.common_update_samples = 0
        self.common_feedforward_target_v = 0.0
        self.common_feedforward_filtered_v = 0.0

        self._validate_mixed_configuration()
        super().__init__(args)

    def _validate_mixed_configuration(self):
        resistance_values = (
            self.mixed_resistance_initial_min,
            self.mixed_resistance_initial_max,
            self.mixed_resistance_target_min,
            self.mixed_resistance_target_max,
        )
        if min(resistance_values) <= 0.0:
            raise ValueError("mixed-load resistance limits must be positive")
        if self.mixed_resistance_initial_min > self.mixed_resistance_initial_max:
            raise ValueError("initial mixed-load resistance limits are reversed")
        if self.mixed_resistance_target_min > self.mixed_resistance_target_max:
            raise ValueError("target mixed-load resistance limits are reversed")
        if self.mixed_cpl_initial_max_w < 0.0 or self.mixed_cpl_target_max_w <= 0.0:
            raise ValueError("mixed-load CPL limits must be non-negative")
        if self.mixed_total_load_power_max_w <= 0.0:
            raise ValueError("dc_mixed_total_load_power_max_w must be positive")
        if not 1 <= self.mixed_event_bus_count_min <= 4:
            raise ValueError("dc_mixed_event_bus_count_min must be in [1, 4]")
        if not self.mixed_event_bus_count_min <= self.mixed_event_bus_count_max <= 4:
            raise ValueError("dc_mixed_event_bus_count_max must be in [min, 4]")
        probabilities = np.asarray(
            [
                self.mixed_resistive_probability,
                self.mixed_cpl_probability,
                self.mixed_combined_probability,
            ],
            dtype=np.float64,
        )
        if np.any(probabilities < 0.0) or float(np.sum(probabilities)) <= 0.0:
            raise ValueError("mixed-load scenario probabilities are invalid")
        self.mixed_scenario_probabilities = probabilities / np.sum(probabilities)
        if not 0.0 <= self.mixed_ramp_probability <= 1.0:
            raise ValueError("dc_mixed_ramp_probability must be in [0, 1]")
        if not 1 <= self.mixed_ramp_steps_min <= self.mixed_ramp_steps_max:
            raise ValueError("mixed-load ramp-step limits are invalid")
        if not 0.0 < self.fast_common_action_filter_alpha <= 1.0:
            raise ValueError("dc_fast_common_action_filter_alpha must be in (0, 1]")
        if self.common_current_feedforward_gain < 0.0:
            raise ValueError("dc_common_current_feedforward_gain must be non-negative")
        if not 0.0 < self.common_feedforward_filter_alpha <= 1.0:
            raise ValueError(
                "dc_common_feedforward_filter_alpha must be in (0, 1]"
            )
        if self.mixed_average_nadir_v <= 0.0:
            raise ValueError("dc_mixed_average_nadir_v must be positive")
        if self.mixed_average_zenith_v <= 48.0:
            raise ValueError("dc_mixed_average_zenith_v must exceed nominal voltage")
        if self.mixed_average_nadir_v >= 48.0:
            raise ValueError("dc_mixed_average_nadir_v must be below nominal voltage")
        if min(
            self.w_mixed_average_nadir,
            self.w_mixed_average_zenith,
            self.w_mixed_transient_iae,
            self.w_mixed_duty_saturation,
            self.w_mixed_common_saturation,
        ) < 0.0:
            raise ValueError("mixed-load reward weights must be non-negative")

    def _curriculum_severity(self):
        progress = min(
            1.0,
            max(0.0, self._mixed_reset_count / float(self.mixed_curriculum_episodes)),
        )
        self.mixed_curriculum_progress = progress
        # Do not start from a trivial plant: the warm-start policy immediately
        # sees at least 35% of the requested operating envelope.
        return 0.35 + 0.65 * min(1.0, progress / 0.80)

    def _limited_cpl(self, resistances, powers):
        powers = np.maximum(np.asarray(powers, dtype=np.float64), 0.0)
        resistive_power = float(
            np.sum((self.nominal_voltage ** 2) / np.asarray(resistances))
        )
        available = max(0.0, self.mixed_total_load_power_max_w - resistive_power)
        total_cpl = float(np.sum(powers))
        if total_cpl > available and total_cpl > 0.0:
            powers *= available / total_cpl
        return powers

    def _sample_mixed_scenario(self):
        severity = self._curriculum_severity()
        self.mixed_scenario_kind = str(
            self.rng.choice(
                np.asarray(["resistive", "cpl", "mixed"], dtype=object),
                p=self.mixed_scenario_probabilities,
            )
        )

        allowed_bus_max = 1 if self.mixed_curriculum_progress < 0.50 else self.mixed_event_bus_count_max
        bus_count = int(
            self.rng.randint(
                self.mixed_event_bus_count_min,
                max(self.mixed_event_bus_count_min, allowed_bus_max) + 1,
            )
        )
        event_buses = self.rng.choice(4, size=bus_count, replace=False)
        self.mixed_event_bus_mask = np.zeros(4, dtype=bool)
        self.mixed_event_bus_mask[event_buses] = True

        initial_r = self.rng.uniform(
            self.mixed_resistance_initial_min,
            self.mixed_resistance_initial_max,
            4,
        )
        candidate_target_r = self.rng.uniform(
            self.mixed_resistance_target_min,
            self.mixed_resistance_target_max,
            4,
        )
        target_r = initial_r.copy()
        if self.mixed_scenario_kind in ("resistive", "mixed"):
            target_r[self.mixed_event_bus_mask] = (
                initial_r[self.mixed_event_bus_mask]
                + severity
                * (
                    candidate_target_r[self.mixed_event_bus_mask]
                    - initial_r[self.mixed_event_bus_mask]
                )
            )

        initial_p = np.zeros(4, dtype=np.float64)
        target_p = np.zeros(4, dtype=np.float64)
        if self.mixed_scenario_kind in ("cpl", "mixed"):
            initial_cap = severity * self.mixed_cpl_initial_max_w
            target_cap = max(100.0, severity * self.mixed_cpl_target_max_w)
            initial_p = self.rng.uniform(0.0, initial_cap, 4)
            target_p = initial_p.copy()
            target_p[self.mixed_event_bus_mask] = self.rng.uniform(
                0.0,
                target_cap,
                int(np.sum(self.mixed_event_bus_mask)),
            )
            minimum_change = min(50.0, 0.25 * target_cap)
            for bus in event_buses:
                if abs(target_p[bus] - initial_p[bus]) < minimum_change:
                    target_p[bus] = (
                        target_cap if initial_p[bus] < 0.5 * target_cap else 0.0
                    )

        self.initial_load_resistances = np.clip(
            initial_r,
            self.mixed_resistance_target_min,
            self.mixed_resistance_initial_max,
        )
        self.target_load_resistances = np.clip(
            target_r,
            self.mixed_resistance_target_min,
            self.mixed_resistance_target_max,
        )
        self.initial_cpl_powers_w = self._limited_cpl(
            self.initial_load_resistances, initial_p
        )
        self.target_cpl_powers_w = self._limited_cpl(
            self.target_load_resistances, target_p
        )

        if self.rng.uniform() < self.mixed_ramp_probability:
            self.active_load_ramp_steps = int(
                self.rng.randint(
                    self.mixed_ramp_steps_min,
                    self.mixed_ramp_steps_max + 1,
                )
            )
        else:
            self.active_load_ramp_steps = 1

        self.active_load_step_agent = int(event_buses[0])
        self.active_cpl_power_w = float(np.max(self.target_cpl_powers_w))

    def _rebuild_initial_state(self):
        self.active_load_step_agent = int(
            np.flatnonzero(self.mixed_event_bus_mask)[0]
        )
        self.load_resistances = self.initial_load_resistances.copy()
        self.cpl_powers_w = self.initial_cpl_powers_w.copy()
        self._refresh_bus_solver()
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

        initial_common_v = float(
            np.clip(
                np.mean(self.droop_coefficients * self.inductor_currents),
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )
        self.common_feedforward_target_v = float(
            np.clip(
                self.common_current_feedforward_gain * initial_common_v,
                0.0,
                self.common_voltage_max_v,
            )
        )
        self.common_feedforward_filtered_v = self.common_feedforward_target_v
        self.common_voltage_integral_v = float(
            np.clip(
                initial_common_v - self.common_feedforward_filtered_v,
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )
        self.common_voltage_action_v = initial_common_v
        self.common_voltage_filtered_v = initial_common_v
        self.actor_action_mean_v = 0.0
        self.actor_differential_actions = np.zeros(4, dtype=np.float64)
        self.actor_differential_filtered_actions = np.zeros(4, dtype=np.float64)
        self.secondary_actions = np.full(4, initial_common_v, dtype=np.float64)
        self.raw_secondary_actions = self.secondary_actions.copy()
        self.previous_secondary_actions = self.secondary_actions.copy()
        self.command_references = np.ones(4) * self.nominal_voltage
        self.voltage_references = self.converter_voltages.copy()
        self._reset_interval_voltage_extrema()

    def reset(self):
        self.cpl_powers_w = np.zeros(4, dtype=np.float64)
        self.pre_step_cpl_powers_w = None
        self._mixed_reset_count += 1
        # Sample the physical scenario before reset() creates a noisy
        # observation.  Consequently normal and noise-free evaluations use
        # the same loads for the same seed.
        self._sample_mixed_scenario()
        DCMicrogridModel.reset(self)
        self._rebuild_initial_state()
        return self._get_obs()

    @staticmethod
    def _event_fraction(t, start, ramp_steps):
        if t < start:
            return 0.0
        if ramp_steps <= 1:
            return 1.0
        return min(1.0, (t - start + 1) / float(ramp_steps))

    def _update_load(self):
        up_fraction = self._event_fraction(
            self.t, self.active_load_step_time, self.active_load_ramp_steps
        )
        if self.t < self.active_load_restore_time:
            active_fraction = up_fraction
        else:
            restore_fraction = self._event_fraction(
                self.t, self.active_load_restore_time, self.active_load_ramp_steps
            )
            active_fraction = 1.0 - restore_fraction

        new_resistances = (
            self.initial_load_resistances
            + active_fraction
            * (self.target_load_resistances - self.initial_load_resistances)
        )
        new_cpl_powers = (
            self.initial_cpl_powers_w
            + active_fraction
            * (self.target_cpl_powers_w - self.initial_cpl_powers_w)
        )

        resistance_changed = not np.allclose(
            new_resistances, self.load_resistances, rtol=0.0, atol=1.0e-12
        )
        self.load_resistances = new_resistances
        self.cpl_powers_w = new_cpl_powers
        if resistance_changed:
            self._refresh_bus_solver()

        self.load_event_this_step = self.t in (
            self.active_load_step_time,
            self.active_load_restore_time,
        )
        if self.load_event_this_step:
            self.last_event_step = self.t
            self.last_transient_event_step = self.t

    def _apply_secondary_action(self, raw_actions):
        """Update only the slow zero-mean sharing action."""
        actor_raw = np.asarray(raw_actions, dtype=np.float64).reshape(4, -1)[:, 0]
        self.actor_action_mean_v = float(np.mean(actor_raw))
        self.actor_differential_actions = actor_raw - self.actor_action_mean_v
        self.actor_differential_filtered_actions += self.action_filter_alpha * (
            self.actor_differential_actions
            - self.actor_differential_filtered_actions
        )
        self.previous_secondary_actions = self.secondary_actions.copy()
        self._compose_secondary_action()

    def _compose_secondary_action(self):
        self.raw_secondary_actions = (
            self.common_voltage_action_v + self.actor_differential_actions
        )
        self.secondary_actions = (
            self.common_voltage_filtered_v
            + self.actor_differential_filtered_actions
        )

    def _update_fast_common_voltage(self):
        average_voltage_v = float(np.mean(self.bus_voltages))
        voltage_error_v = self.nominal_voltage - average_voltage_v
        update_dt = self.voltage_dt if self.fast_common_voltage_enabled else self.secondary_dt
        estimated_droop_v = float(
            np.mean(self.droop_coefficients * self.inductor_currents)
        )
        self.common_feedforward_target_v = float(
            np.clip(
                self.common_current_feedforward_gain * estimated_droop_v,
                0.0,
                self.common_voltage_max_v,
            )
        )
        self.common_feedforward_filtered_v += (
            self.common_feedforward_filter_alpha
            * (
                self.common_feedforward_target_v
                - self.common_feedforward_filtered_v
            )
        )
        candidate_integral = float(
            np.clip(
                self.common_voltage_integral_v
                + self.common_voltage_ki * voltage_error_v * update_dt,
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )
        unsaturated = (
            self.common_voltage_kp * voltage_error_v
            + candidate_integral
            + self.common_feedforward_filtered_v
        )
        saturated = float(
            np.clip(
                unsaturated,
                self.common_voltage_min_v,
                self.common_voltage_max_v,
            )
        )
        can_integrate = (
            np.isclose(unsaturated, saturated)
            or (saturated >= self.common_voltage_max_v and voltage_error_v < 0.0)
            or (saturated <= self.common_voltage_min_v and voltage_error_v > 0.0)
        )
        if can_integrate:
            self.common_voltage_integral_v = candidate_integral
        self.common_voltage_action_v = saturated
        filter_alpha = (
            self.fast_common_action_filter_alpha
            if self.fast_common_voltage_enabled
            else self.common_action_filter_alpha
        )
        self.common_voltage_filtered_v += filter_alpha * (
            self.common_voltage_action_v - self.common_voltage_filtered_v
        )
        self.common_update_samples += 1
        self.common_saturation_samples += int(not np.isclose(unsaturated, saturated))
        self._compose_secondary_action()

    def _run_multirate_controller(self):
        if not self.fast_common_voltage_enabled:
            # Preserve the historical 100 Hz path if explicitly requested.
            self._update_fast_common_voltage()
            super()._run_multirate_controller()
            self.last_action_change = float(
                np.mean(np.abs(self.secondary_actions - self.previous_secondary_actions))
            )
            return

        self._reset_interval_voltage_extrema()
        voltage_count = 0
        for current_step in range(self.current_steps_per_secondary):
            if voltage_count == 0:
                self._update_fast_common_voltage()
            self._update_voltage_reference()
            if voltage_count == 0:
                self._voltage_outer_loop()
            self._current_inner_loop()
            self._plant_step(
                record_pre_update=(self.load_event_this_step and current_step == 0)
            )
            voltage_count += 1
            if voltage_count >= self.current_steps_per_voltage:
                voltage_count = 0
        self.last_action_change = float(
            np.mean(np.abs(self.secondary_actions - self.previous_secondary_actions))
        )

    def _reset_interval_voltage_extrema(self):
        super()._reset_interval_voltage_extrema()
        self.interval_average_abs_error_sum_v = 0.0
        self.interval_voltage_sample_count = 0
        self.common_saturation_samples = 0
        self.common_update_samples = 0

    def _record_interval_voltage_extrema(self):
        super()._record_interval_voltage_extrema()
        self.interval_average_abs_error_sum_v += abs(
            float(np.mean(self.bus_voltages)) - self.nominal_voltage
        )
        self.interval_voltage_sample_count += 1

    def _reward(self):
        reward, info = super()._reward()
        minimum_average_voltage_v = float(self.interval_average_voltage_min_v)
        maximum_average_voltage_v = float(self.interval_average_voltage_max_v)
        average_nadir_excess = max(
            0.0,
            (self.mixed_average_nadir_v - minimum_average_voltage_v)
            / self.mixed_average_nadir_v,
        )
        average_zenith_excess = max(
            0.0,
            (maximum_average_voltage_v - self.mixed_average_zenith_v)
            / self.mixed_average_zenith_v,
        )
        mean_transient_error_pu = (
            self.interval_average_abs_error_sum_v
            / max(1, self.interval_voltage_sample_count)
            / self.nominal_voltage
        )
        transient_iae_penalty = (
            mean_transient_error_pu ** 2
            if float(info.get("transient_active", 0.0)) > 0.5
            else 0.0
        )
        common_saturation_fraction = (
            self.common_saturation_samples / max(1, self.common_update_samples)
        )
        mixed_penalty = (
            self.w_mixed_average_nadir * average_nadir_excess ** 2
            + self.w_mixed_average_zenith * average_zenith_excess ** 2
            + self.w_mixed_transient_iae * transient_iae_penalty
            + self.w_mixed_duty_saturation
            * float(info["duty_saturation_fraction"]) ** 2
            + self.w_mixed_common_saturation * common_saturation_fraction ** 2
        )
        reward -= mixed_penalty
        total_load_power_w = float(
            np.sum(self.load_currents * self.bus_voltages)
        )
        info.update(
            {
                "load_model": "random_resistive_cpl_mixed",
                "mixed_scenario_kind": self.mixed_scenario_kind,
                "mixed_curriculum_progress": float(self.mixed_curriculum_progress),
                "mixed_event_bus_count": int(np.sum(self.mixed_event_bus_mask)),
                "mixed_ramp_steps": int(self.active_load_ramp_steps),
                "total_load_power_w": total_load_power_w,
                "mixed_minimum_average_voltage_v": minimum_average_voltage_v,
                "mixed_maximum_average_voltage_v": maximum_average_voltage_v,
                "mixed_average_nadir_penalty": float(average_nadir_excess ** 2),
                "mixed_average_zenith_penalty": float(average_zenith_excess ** 2),
                "mixed_transient_iae_penalty": float(transient_iae_penalty),
                "common_feedforward_target_v": float(
                    self.common_feedforward_target_v
                ),
                "common_feedforward_filtered_v": float(
                    self.common_feedforward_filtered_v
                ),
                "common_saturation_fraction": float(common_saturation_fraction),
                "weighted_mixed_penalty": float(mixed_penalty),
            }
        )
        for agent_id in range(self.num_agents):
            info[f"initial_R{agent_id + 1}_ohm"] = float(
                self.initial_load_resistances[agent_id]
            )
            info[f"target_R{agent_id + 1}_ohm"] = float(
                self.target_load_resistances[agent_id]
            )
            info[f"initial_P{agent_id + 1}_w"] = float(
                self.initial_cpl_powers_w[agent_id]
            )
            info[f"target_P{agent_id + 1}_w"] = float(
                self.target_cpl_powers_w[agent_id]
            )
            info[f"event_bus_{agent_id + 1}"] = float(
                self.mixed_event_bus_mask[agent_id]
            )
        return float(reward), info

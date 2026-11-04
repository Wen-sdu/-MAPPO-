"""HIL-aligned MAPPO secondary controller training model.

The controller architecture intentionally matches the new deployment contract.
The Actor output is interpreted directly in volts:

    u_i = MAPPO(o_i)
    Vcmd_i = Vnom - kd_i * IL_i + u_i
    voltage PI(Vref_i - Vo_i) -> ILref_i
    current PI(ILref_i - IL_i) -> duty_i

There is no duty feed-forward, actor blending, common-mode removal, auxiliary
current-sharing PI, or auxiliary average-voltage restoration PI.  The actor is
the complete secondary controller.  Vo is the converter-side voltage before
Rs; Vdc is the bus voltage after Rs.
"""

import numpy as np


def _parse_vector(value, count, name, positive=True):
    if isinstance(value, str):
        values = [float(x.strip()) for x in value.replace(";", ",").split(",") if x.strip()]
    else:
        values = np.asarray(value, dtype=np.float64).reshape(-1).tolist()
    if len(values) == 1:
        values *= count
    result = np.asarray(values, dtype=np.float64)
    if result.size != count:
        raise ValueError("{} must contain {} values".format(name, count))
    if positive and np.any(result <= 0.0):
        raise ValueError("{} values must be positive".format(name))
    return result


class DCMicrogridModel(object):
    """Averaged four-buck/ring-bus model with three controller time scales."""

    def __init__(self, args):
        self.num_agents = int(getattr(args, "num_agents", 4))
        if self.num_agents != 4:
            raise ValueError("The current Typhoon model contains exactly four converters")
        self.episode_length = int(getattr(args, "episode_length", 200))
        self.nominal_voltage = 48.0
        self.capacity_ratios = _parse_vector(
            getattr(args, "dc_capacity_ratios", "1,2,1,1"), 4, "dc_capacity_ratios"
        )
        self.capacity_ratios /= np.min(self.capacity_ratios)
        self.capacity_sum = float(np.sum(self.capacity_ratios))

        # Physical parameters copied from Fourbuck_MultiBus.tse.
        self.input_voltage = float(getattr(args, "dc_input_voltage", 100.0))
        self.inductance = float(getattr(args, "dc_inductance", 0.005))
        self.capacitance = float(getattr(args, "dc_capacitance", 0.0022))
        self.source_resistances = _parse_vector(
            getattr(args, "dc_source_resistances", "0.01,0.005,0.008,0.008"),
            4,
            "dc_source_resistances",
        )
        self.line_resistance = float(getattr(args, "dc_line_resistance", 1.0))

        # Exact deployed controller schedule: 10 kHz / 1 kHz / 100 Hz.
        self.current_dt = float(getattr(args, "dc_current_loop_dt", 0.0001))
        self.voltage_dt = float(getattr(args, "dc_voltage_loop_dt", 0.001))
        self.secondary_dt = float(getattr(args, "dc_secondary_loop_dt", 0.010))
        self.current_steps_per_secondary = int(round(self.secondary_dt / self.current_dt))
        self.current_steps_per_voltage = int(round(self.voltage_dt / self.current_dt))
        if self.current_steps_per_secondary <= 0 or self.current_steps_per_voltage <= 0:
            raise ValueError("Controller periods must be positive")
        if not np.isclose(
            self.current_steps_per_secondary * self.current_dt, self.secondary_dt
        ):
            raise ValueError("secondary period must be an integer multiple of current period")
        if not np.isclose(self.current_steps_per_voltage * self.current_dt, self.voltage_dt):
            raise ValueError("voltage period must be an integer multiple of current period")

        # Primary droop k_i = k_base / capacity_i implements 1:2:1:1 sharing.
        self.droop_base = float(getattr(args, "dc_droop_base_v_per_a", 1.0))
        self.droop_coefficients = self.droop_base / self.capacity_ratios

        # The unbounded Gaussian Actor output is the secondary correction u_i in volts.

        # Cascaded PI and safety limits.
        self.voltage_pi_kp = float(getattr(args, "dc_voltage_pi_kp", 0.30))
        self.voltage_pi_ki = float(getattr(args, "dc_voltage_pi_ki", 25.0))
        self.current_pi_kp = float(getattr(args, "dc_current_pi_kp", 0.12))
        self.current_pi_ki = float(getattr(args, "dc_current_pi_ki", 0.25))
        # Keep only the physical Buck PWM range;
        # engineering guard band.
        self.duty_min = float(getattr(args, "dc_duty_min", 0.0))
        self.duty_max = float(getattr(args, "dc_duty_max", 1.0))

        # Physical 40 ohm base loads; bus 2 steps to 30 ohm and later returns to 40 ohm.
        self.load_resistance_min = float(getattr(args, "dc_load_resistance_min", 30.0))
        self.load_resistance_max = float(getattr(args, "dc_load_resistance_max", 40.0))
        self.initial_load_min = float(
            getattr(args, "dc_initial_load_resistance_min", 40.0) or 40.0
        )
        self.initial_load_max = float(
            getattr(args, "dc_initial_load_resistance_max", 40.0) or 40.0
        )
        self.load_step_time = int(getattr(args, "dc_load_step_time", 30))
        self.load_restore_time = int(getattr(args, "dc_load_restore_time", 100))
        self.randomize_load_times = bool(
            getattr(args, "dc_randomize_load_times", False)
        )
        self.load_step_time_min = int(
            getattr(args, "dc_load_step_time_min", 20)
        )
        self.load_step_time_max = int(
            getattr(args, "dc_load_step_time_max", 60)
        )
        self.load_step_duration_min = int(
            getattr(args, "dc_load_step_duration_min", 60)
        )
        self.load_step_duration_max = int(
            getattr(args, "dc_load_step_duration_max", 80)
        )
        self.load_step_agent = int(getattr(args, "dc_load_step_agent", 1))
        self.randomize_load_step_agent = bool(#是否随机选择发生负载突变的母线
            getattr(args, "dc_randomize_load_step_agent", False)
        )
        step_min = getattr(args, "dc_load_step_target_min", 30.0)
        step_max = getattr(args, "dc_load_step_target_max", 30.0)
        self.load_step_target_min = float(30.0 if step_min is None else step_min)
        self.load_step_target_max = float(30.0 if step_max is None else step_max)
        self.load_random_walk = float(getattr(args, "dc_load_random_walk", 0.0))

        # Observation and measurement-domain randomisation.
        self.current_base = float(getattr(args, "dc_current_base", 10.0))
        self.obs_voltage_delta = float(getattr(args, "dc_obs_voltage_delta_v", 9.6))
        self.voltage_noise_std = float(getattr(args, "dc_noise_std", 0.005))
        self.current_noise_std = float(getattr(args, "dc_current_noise_std", 0.01))
        self.action_filter_alpha = float(
            getattr(args, "dc_action_filter_alpha", 1.0)
        )
        if not 0.0 < self.action_filter_alpha <= 1.0:
            raise ValueError("dc_action_filter_alpha must be in (0, 1]")

        # Convergence targets and the three baseline reward weights.
        self.average_voltage_band_v = float(
            getattr(args, "dc_average_voltage_band_v", 0.5)
        )
        self.voltage_safe_fraction = float(getattr(args, "dc_voltage_safe_error", 0.05))
        self.current_target_error = float(getattr(args, "dc_current_safe_error", 0.02))
        self.settling_steps = max(1, int(getattr(args, "dc_settling_steps", 20)))
        self.stability_window_steps = max(
            5, int(getattr(args, "dc_stability_window_steps", 50))
        )
        self.w_average_voltage = float(getattr(args, "dc_reward_average_voltage", 14.0))
        self.w_current_share = float(getattr(args, "dc_reward_current_share", 12.0))
        self.w_settling = float(getattr(args, "dc_settling_penalty", 12.0))

        # Steady-state temporal penalties.  Current sharing alone cannot see a
        # common proportional oscillation (all IL_i/c_i moving together), so
        # explicitly penalise both secondary-action movement and the temporal
        # change of capacity-normalised current.  Zero weights preserve old
        # checkpoints and historical RMAPPO training behaviour.
        self.w_action_change = float(
            getattr(args, "dc_reward_action_change", 0.0)
        )
        self.w_current_ripple = float(
            getattr(args, "dc_reward_current_ripple", 0.0)
        )
        self.action_change_scale_v = float(
            getattr(args, "dc_action_change_scale_v", 0.10)
        )
        self.current_ripple_scale_a = float(
            getattr(args, "dc_current_ripple_scale_a", 0.05)
        )
        if self.w_action_change < 0.0 or self.w_current_ripple < 0.0:
            raise ValueError("steady-state reward weights must be non-negative")
        if self.action_change_scale_v <= 0.0:
            raise ValueError("dc_action_change_scale_v must be positive")
        if self.current_ripple_scale_a <= 0.0:
            raise ValueError("dc_current_ripple_scale_a must be positive")

        # Optional event-window penalties use the extrema observed inside every
        # 10 ms secondary-control interval.  Keeping both weights at zero
        # exactly preserves the baseline reward used to train run3.
        self.transient_window_steps = max(
            0, int(getattr(args, "dc_transient_window_steps", 10))
        )
        self.transient_average_voltage_band_v = float(
            getattr(args, "dc_transient_average_voltage_band_v", 0.25)
        )
        self.transient_bus_voltage_band_v = float(
            getattr(args, "dc_transient_bus_voltage_band_v", 2.4)
        )
        self.w_transient_average_voltage = float(
            getattr(args, "dc_reward_transient_average_voltage", 0.0)
        )
        self.w_transient_bus_voltage = float(
            getattr(args, "dc_reward_transient_bus_voltage", 0.0)
        )
        if self.transient_average_voltage_band_v <= 0.0:
            raise ValueError("dc_transient_average_voltage_band_v must be positive")
        if self.transient_bus_voltage_band_v <= 0.0:
            raise ValueError("dc_transient_bus_voltage_band_v must be positive")
        if self.w_transient_average_voltage < 0.0 or self.w_transient_bus_voltage < 0.0:
            raise ValueError("transient reward weights must be non-negative")

        self.adjacency = self._ring_adjacency()
        self.rng = np.random.RandomState(1)
        self.reset()

    @property
    def voltages(self):
        """Compatibility alias: report actual bus-side Vdc values."""
        return self.bus_voltages

    @property
    def branch_currents(self):
        """Compatibility alias: IL1~IL4 are the configured current signals."""
        return self.inductor_currents

    @property
    def secondary_voltage_actions(self):
        """Secondary-voltage actions applied directly in volts."""
        return self.secondary_actions

    def seed(self, seed=None):
        self.rng = np.random.RandomState(1 if seed is None else int(seed))

    def _ring_adjacency(self):
        adjacency = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            adjacency[i, (i - 1) % 4] = 1.0
            adjacency[i, (i + 1) % 4] = 1.0
        return adjacency

    def _refresh_bus_solver(self):#构造电导矩阵，并求该电导矩阵的逆矩阵
        conductance = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            conductance[i, i] = (
                1.0 / self.source_resistances[i]
                + 1.0 / max(self.load_resistances[i], 1e-6)
                + np.sum(self.adjacency[i]) / self.line_resistance
            )
            for j in np.where(self.adjacency[i] > 0.0)[0]:
                conductance[i, j] = -1.0 / self.line_resistance
        self._bus_inverse = np.linalg.inv(conductance)

    def _solve_bus_voltages(self):#根据变换器输出电压，求解四条母线实际电压
        rhs = self.converter_voltages / self.source_resistances
        return np.matmul(self._bus_inverse, rhs)

    def reset(self):
        self.t = 0
        if self.randomize_load_times:
            step_low = min(self.load_step_time_min, self.load_step_time_max)
            step_high = max(self.load_step_time_min, self.load_step_time_max)
            duration_low = min(
                self.load_step_duration_min, self.load_step_duration_max
            )
            duration_high = max(
                self.load_step_duration_min, self.load_step_duration_max
            )
            latest_step = min(step_high, self.episode_length - duration_low - 1)
            if step_low < 1 or duration_low < 1 or latest_step < step_low:
                raise ValueError(
                    "Random load-event ranges do not fit inside the episode"
                )
            self.active_load_step_time = int(
                self.rng.randint(step_low, latest_step + 1)
            )
            latest_duration = min(
                duration_high,
                self.episode_length - self.active_load_step_time - 1,
            )
            self.active_load_restore_time = self.active_load_step_time + int(
                self.rng.randint(duration_low, latest_duration + 1)
            )
        else:
            self.active_load_step_time = self.load_step_time
            self.active_load_restore_time = self.load_restore_time
        self.load_resistances = self.rng.uniform(self.initial_load_min, self.initial_load_max, 4)
        self._refresh_bus_solver()
        self.converter_voltages = self.nominal_voltage + self.rng.uniform(-0.05, 0.05, 4)#输出电压初始化
        self.bus_voltages = self._solve_bus_voltages()
        self.output_currents = (#输出电流
            self.converter_voltages - self.bus_voltages
        ) / self.source_resistances
        self.inductor_currents = np.maximum(self.output_currents, 0.0)
        self.load_currents = self.bus_voltages / self.load_resistances
        self.secondary_actions = np.zeros(4, dtype=np.float64)#初始化4个Agent二次控制ui为0
        self.raw_secondary_actions = np.zeros(4, dtype=np.float64)
        self.previous_secondary_actions = np.zeros(4, dtype=np.float64)
        self.previous_per_capacity_currents = (
            self.inductor_currents / self.capacity_ratios
        ).copy()
        self.command_references = np.ones(4) * self.nominal_voltage
        self.voltage_references = self.converter_voltages.copy()
        self.current_references = self.inductor_currents.copy()
        self.voltage_pi_integral = self.current_references.copy()#初始电压环PI积分项
        # This is only an initial controller state for a pre-energised episode;
        # no Vout/Vin term is added to the current PI during operation.
        self.current_pi_integral = np.clip(#初始电流环PI积分项
            self.converter_voltages / self.input_voltage, self.duty_min, self.duty_max
        )
        self.duty = self.current_pi_integral.copy()
        self.last_action_change = 0.0
        self.last_event_step = 0
        self.last_transient_event_step = None
        self.load_event_this_step = False
        self.pre_step_load_resistances = None
        self.active_load_step_agent = (
            int(self.rng.randint(0, 4)) if self.randomize_load_step_agent else self.load_step_agent
        )
        target_low = min(self.load_step_target_min, self.load_step_target_max)
        target_high = max(self.load_step_target_min, self.load_step_target_max)
        self.active_load_step_target = float(
            target_low
            if np.isclose(target_low, target_high)
            else self.rng.uniform(target_low, target_high)
        )
        self.voltage_history = []
        self.current_history = []#电压、电流的历史数据缓冲区
        self.reference_saturation_fraction = 0.0
        self.duty_saturation_fraction = 0.0
        self._reset_interval_voltage_extrema()
        return self._get_obs()

    def _update_load(self):
        self.load_event_this_step = False
        if self.load_random_walk > 0.0:
            self.load_resistances += self.rng.normal(0.0, self.load_random_walk, 4)
        if self.t == self.active_load_step_time:
            self.pre_step_load_resistances = self.load_resistances.copy()
            self.load_resistances[self.active_load_step_agent] = self.active_load_step_target
            self.last_event_step = self.t
            self.last_transient_event_step = self.t
            self.load_event_this_step = True
        if (
            self.t == self.active_load_restore_time
            and self.pre_step_load_resistances is not None
        ):
            self.load_resistances = self.pre_step_load_resistances.copy()
            self.pre_step_load_resistances = None
            self.last_event_step = self.t
            self.last_transient_event_step = self.t
            self.load_event_this_step = True
        self.load_resistances = np.clip(
            self.load_resistances, self.load_resistance_min, self.load_resistance_max
        )
        self._refresh_bus_solver()

    def _apply_secondary_action(self, raw_actions):#更新二次控制信号
        raw = np.asarray(raw_actions, dtype=np.float64).reshape(4, -1)[:, 0]
        self.raw_secondary_actions = raw.copy()
        self.previous_secondary_actions = self.secondary_actions.copy()
        self.secondary_actions += self.action_filter_alpha * (
            raw - self.secondary_actions
        )
        self.last_action_change = float(
            np.mean(np.abs(self.secondary_actions - self.previous_secondary_actions))
        )

    def _update_voltage_reference(self):
        raw_command = (
            self.nominal_voltage
            - self.droop_coefficients * self.inductor_currents
            + self.secondary_actions
        )
        self.command_references = raw_command
        self.voltage_references = self.command_references.copy()

    def _voltage_outer_loop(self):#电压环PI
        error = self.voltage_references - self.converter_voltages
        self.voltage_pi_integral += self.voltage_pi_ki * error * self.voltage_dt
        self.current_references = (
            self.voltage_pi_kp * error + self.voltage_pi_integral
        )

    def _current_inner_loop(self):#电流环PI
        error = self.current_references - self.inductor_currents
        proposed = self.current_pi_integral + self.current_pi_ki * error * self.current_dt
        unsaturated = self.current_pi_kp * error + proposed
        duty = np.clip(unsaturated, self.duty_min, self.duty_max)
        integrate = (
            np.isclose(unsaturated, duty)
            | ((duty <= self.duty_min) & (error > 0.0))
            | ((duty >= self.duty_max) & (error < 0.0))
        )
        self.current_pi_integral[integrate] = proposed[integrate]
        self.current_pi_integral = np.clip(
            self.current_pi_integral, self.duty_min, self.duty_max
        )
        self.duty = duty
        self.duty_saturation_fraction = float(np.mean(~integrate))

    def _plant_step(self, record_pre_update=False):
        # Explicit averaged buck update at the real 10 kHz controller period.
        di = (
            self.duty * self.input_voltage - self.converter_voltages
        ) / self.inductance
        self.inductor_currents += self.current_dt * di
        self.inductor_currents = np.maximum(self.inductor_currents, 0.0)

        self.bus_voltages = self._solve_bus_voltages()
        if record_pre_update:
            # This extra point captures the algebraic jump immediately after a
            # load event.  On ordinary substeps it would only duplicate the
            # preceding capacitor-update endpoint.
            self._record_interval_voltage_extrema()
        self.output_currents = (
            self.converter_voltages - self.bus_voltages
        ) / self.source_resistances
        dv = (
            self.inductor_currents - self.output_currents
        ) / self.capacitance
        self.converter_voltages += self.current_dt * dv
        self.bus_voltages = self._solve_bus_voltages()
        self._record_interval_voltage_extrema()
        self.output_currents = (
            self.converter_voltages - self.bus_voltages
        ) / self.source_resistances
        self.load_currents = self.bus_voltages / self.load_resistances

    def _reset_interval_voltage_extrema(self):
        """Start a new 10 ms interval without changing the simulated state."""
        self.interval_bus_voltage_max_v = np.full(self.num_agents, -np.inf)
        self.interval_bus_voltage_min_v = np.full(self.num_agents, np.inf)
        self.interval_average_voltage_max_v = -np.inf
        self.interval_average_voltage_min_v = np.inf

    def _record_interval_voltage_extrema(self):
        """Accumulate extrema at the 0.1 ms plant-model resolution."""
        self.interval_bus_voltage_max_v = np.maximum(
            self.interval_bus_voltage_max_v, self.bus_voltages
        )
        self.interval_bus_voltage_min_v = np.minimum(
            self.interval_bus_voltage_min_v, self.bus_voltages
        )
        average_voltage = float(np.mean(self.bus_voltages))
        self.interval_average_voltage_max_v = max(
            self.interval_average_voltage_max_v, average_voltage
        )
        self.interval_average_voltage_min_v = min(
            self.interval_average_voltage_min_v, average_voltage
        )

    def _run_multirate_controller(self):
        self._reset_interval_voltage_extrema()
        voltage_count = 0
        for current_step in range(self.current_steps_per_secondary):
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

    def _sharing_metrics(self):
        per_capacity = self.inductor_currents / self.capacity_ratios
        share_lambda = float(np.mean(per_capacity))
        denominator = max(abs(share_lambda), 0.1 * self.current_base, 1e-6)
        deviations = np.abs(per_capacity - share_lambda) / denominator
        return share_lambda, per_capacity, float(np.mean(deviations)), float(np.max(deviations))

    def _stability_metrics(self, per_capacity):
        self.voltage_history.append(float(np.mean(self.bus_voltages)))
        self.current_history.append(np.asarray(per_capacity).copy())
        self.voltage_history = self.voltage_history[-self.stability_window_steps :]
        self.current_history = self.current_history[-self.stability_window_steps :]
        if len(self.voltage_history) < 5:
            return {
                "voltage_slope_abs_v_per_s": 0.0,
                "voltage_detrended_rms_v": 0.0,
                "current_max_slope_abs_a_per_s": 0.0,
                "current_max_detrended_rms_a": 0.0,
            }
        time = np.arange(len(self.voltage_history)) * self.secondary_dt
        time -= np.mean(time)
        energy = max(float(np.dot(time, time)), 1e-12)
        voltage = np.asarray(self.voltage_history)
        voltage -= np.mean(voltage)
        voltage_slope = float(np.dot(time, voltage) / energy)
        voltage_residual = voltage - voltage_slope * time
        current = np.asarray(self.current_history)
        current -= np.mean(current, axis=0)
        current_slopes = np.dot(time, current) / energy
        current_residual = current - np.outer(time, current_slopes)
        return {
            "voltage_slope_abs_v_per_s": abs(voltage_slope),
            "voltage_detrended_rms_v": float(np.sqrt(np.mean(voltage_residual ** 2))),
            "current_max_slope_abs_a_per_s": float(np.max(np.abs(current_slopes))),
            "current_max_detrended_rms_a": float(
                np.max(np.sqrt(np.mean(current_residual ** 2, axis=0)))
            ),
        }

    def _errors(self):
        mean_voltage_error = float(
            np.mean(np.abs(self.bus_voltages - self.nominal_voltage)) / self.nominal_voltage
        )
        average_voltage_error = float(#先求电压平均值与额定电压之差的绝对值，再除以额定电压，平均电压误差。
            abs(np.mean(self.bus_voltages) - self.nominal_voltage) / self.nominal_voltage
        )
        _, _, current_error, _ = self._sharing_metrics()
        return mean_voltage_error, average_voltage_error, current_error

    def _reward(self):
        average_voltage_error_v = abs(float(np.mean(self.bus_voltages)) - self.nominal_voltage)
        average_voltage_error = average_voltage_error_v / self.nominal_voltage
        max_voltage_error = float(
            np.max(np.abs(self.bus_voltages - self.nominal_voltage)) / self.nominal_voltage
        )
        share_lambda, per_capacity, current_error, max_current_error = self._sharing_metrics()
        stability = self._stability_metrics(per_capacity)
        steps_since_event = self.t - self.last_event_step
        steady = steps_since_event >= self.settling_steps

        action_delta_v = self.secondary_actions - self.previous_secondary_actions
        current_delta_a = per_capacity - self.previous_per_capacity_currents
        action_delta_rms_v = float(np.sqrt(np.mean(action_delta_v ** 2)))
        current_delta_rms_a = float(np.sqrt(np.mean(current_delta_a ** 2)))
        # The ripple objectives are intentionally disabled during the settling
        # window so the policy is not punished for the necessary load-step
        # response.  They become active once steady operation is expected.
        steady_action_change_penalty = 0.0
        steady_current_ripple_penalty = 0.0
        if steady:
            steady_action_change_penalty = float(
                np.mean((action_delta_v / self.action_change_scale_v) ** 2)
            )
            steady_current_ripple_penalty = float(
                np.mean((current_delta_a / self.current_ripple_scale_a) ** 2)
            )
        weighted_steady_stability_penalty = (
            self.w_action_change * steady_action_change_penalty
            + self.w_current_ripple * steady_current_ripple_penalty
        )
        settling_violation = 0.0
        if steady:
            if average_voltage_error_v > self.average_voltage_band_v:
                settling_violation += average_voltage_error_v / self.average_voltage_band_v
            if current_error > self.current_target_error:
                settling_violation += (
                    current_error - self.current_target_error
                ) / max(self.current_target_error, 1e-6)
        average_voltage_peak_deviation_v = max(
            abs(self.interval_average_voltage_max_v - self.nominal_voltage),
            abs(self.interval_average_voltage_min_v - self.nominal_voltage),
        )
        bus_voltage_peak_deviations_v = np.maximum(
            np.abs(self.interval_bus_voltage_max_v - self.nominal_voltage),
            np.abs(self.interval_bus_voltage_min_v - self.nominal_voltage),
        )
        max_bus_voltage_peak_deviation_v = float(
            np.max(bus_voltage_peak_deviations_v)
        )
        if self.last_transient_event_step is None:
            transient_steps_since_event = -1
            transient_active = False
        else:
            transient_steps_since_event = self.t - self.last_transient_event_step
            transient_active = (
                0 <= transient_steps_since_event < self.transient_window_steps
            )
        transient_average_voltage_penalty = 0.0
        transient_bus_voltage_penalty = 0.0
        if transient_active:
            average_excess = max(
                0.0,
                average_voltage_peak_deviation_v
                - self.transient_average_voltage_band_v,
            ) / self.transient_average_voltage_band_v
            bus_excess = max(
                0.0,
                max_bus_voltage_peak_deviation_v
                - self.transient_bus_voltage_band_v,
            ) / self.transient_bus_voltage_band_v
            transient_average_voltage_penalty = average_excess ** 2
            transient_bus_voltage_penalty = bus_excess ** 2
        weighted_transient_voltage_penalty = (
            self.w_transient_average_voltage * transient_average_voltage_penalty
            + self.w_transient_bus_voltage * transient_bus_voltage_penalty
        )

        reward = (
            -self.w_average_voltage * average_voltage_error
            -self.w_current_share * current_error
            -self.w_settling * settling_violation
            -weighted_transient_voltage_penalty
            -weighted_steady_stability_penalty
        )
        info = {
            "voltage_error": float(np.mean(np.abs(self.bus_voltages - self.nominal_voltage)) / self.nominal_voltage),
            "average_voltage_error": float(average_voltage_error),
            "max_voltage_error": float(max_voltage_error),
            "max_voltage_deviation_v": float(np.max(np.abs(self.bus_voltages - self.nominal_voltage))),
            "voltage_within_safe_band": float(max_voltage_error <= self.voltage_safe_fraction),
            "voltage_safe_low_v": self.nominal_voltage * (1.0 - self.voltage_safe_fraction),
            "voltage_safe_high_v": self.nominal_voltage * (1.0 + self.voltage_safe_fraction),
            "average_voltage_band_pass": float(average_voltage_error_v <= self.average_voltage_band_v),
            "current_error": float(current_error),
            "max_current_error": float(max_current_error),
            "action_change": self.last_action_change,
            "action_delta_rms_v": action_delta_rms_v,
            "current_per_capacity_delta_rms_a": current_delta_rms_a,
            "steady_action_change_penalty": steady_action_change_penalty,
            "steady_current_ripple_penalty": steady_current_ripple_penalty,
            "weighted_steady_stability_penalty": float(
                weighted_steady_stability_penalty
            ),
            "steady_stability_penalty_active": float(steady),
            "action_filter_alpha": self.action_filter_alpha,
            "base_reward": 0.0,
            "stage_count": 0,
            "average_voltage": float(np.mean(self.bus_voltages)),
            "average_converter_voltage": float(np.mean(self.converter_voltages)),
            "average_current": float(np.mean(self.inductor_currents)),
            "average_load": float(np.mean(self.load_currents)),
            "average_pv": 0.0,
            "average_load_resistance": float(np.mean(self.load_resistances)),
            "share_lambda": float(share_lambda),
            "mean_current_share_error_a": float(np.mean(np.abs(per_capacity - share_lambda))),
            "mean_current_per_capacity_error_a": float(np.mean(np.abs(per_capacity - share_lambda))),
            "current_per_capacity_spread_a": float(np.max(per_capacity) - np.min(per_capacity)),
            "steps_since_event": int(steps_since_event),
            "settled_within_0p2s": float(settling_violation == 0.0 if steady else 0.0),
            "load_step_time": int(self.active_load_step_time),
            "load_restore_time": int(self.active_load_restore_time),
            "load_event": float(self.load_event_this_step),
            "transient_active": float(transient_active),
            "transient_steps_since_event": int(transient_steps_since_event),
            "average_voltage_peak_max_v": float(self.interval_average_voltage_max_v),
            "average_voltage_peak_min_v": float(self.interval_average_voltage_min_v),
            "average_voltage_peak_deviation_v": float(average_voltage_peak_deviation_v),
            "max_bus_voltage_peak_deviation_v": max_bus_voltage_peak_deviation_v,
            "peak_voltage_within_safe_band": float(
                max_bus_voltage_peak_deviation_v
                <= self.nominal_voltage * self.voltage_safe_fraction
            ),
            "transient_average_voltage_penalty": float(
                transient_average_voltage_penalty
            ),
            "transient_bus_voltage_penalty": float(transient_bus_voltage_penalty),
            "weighted_transient_voltage_penalty": float(
                weighted_transient_voltage_penalty
            ),
            "duty_saturation_fraction": self.duty_saturation_fraction,
            **stability,
        }
        for agent_id in range(self.num_agents):
            info["V{}_peak_max".format(agent_id)] = float(
                self.interval_bus_voltage_max_v[agent_id]
            )
            info["V{}_peak_min".format(agent_id)] = float(
                self.interval_bus_voltage_min_v[agent_id]
            )
        return float(reward), info

    def step(self, raw_actions):
        self.previous_per_capacity_currents = (
            self.inductor_currents / self.capacity_ratios
        ).copy()
        self.t += 1
        self._update_load()
        self._apply_secondary_action(raw_actions)
        self._run_multirate_controller()
        reward, info = self._reward()
        catastrophic = bool(
            np.any(~np.isfinite(self.bus_voltages))
            or np.any(~np.isfinite(self.interval_bus_voltage_min_v))
            or np.any(~np.isfinite(self.interval_bus_voltage_max_v))
            or np.any(self.interval_bus_voltage_min_v < 20.0)
            or np.any(self.interval_bus_voltage_max_v > 70.0)
        )
        done = self.t >= self.episode_length or catastrophic
        if catastrophic:
            # Keep the rollout buffer finite even if a numerical failure made
            # the normally computed reward NaN or infinite.
            reward = -100.0
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        measured_current = self.inductor_currents.copy()
        measured_voltage = self.bus_voltages.copy()
        if self.current_noise_std > 0.0:
            measured_current += self.rng.normal(0.0, self.current_noise_std, 4)
        if self.voltage_noise_std > 0.0:
            measured_voltage += self.rng.normal(0.0, self.voltage_noise_std, 4)
        per_capacity = measured_current / self.capacity_ratios
        for i in range(4):
            neighbors = [(i - 1) % 4, (i + 1) % 4]
            currents = [per_capacity[i]] + [per_capacity[j] for j in neighbors]
            voltages = [measured_voltage[i]] + [measured_voltage[j] for j in neighbors]
            current_obs = np.asarray(currents) / (2.0 * self.current_base)
            voltage_obs = (
                np.asarray(voltages) - self.nominal_voltage
            ) / self.obs_voltage_delta
            obs.append(np.concatenate([current_obs, voltage_obs]).astype(np.float32))
        return np.asarray(obs, dtype=np.float32)

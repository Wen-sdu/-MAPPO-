import numpy as np

from onpolicy.envs.dc_microgrid.secondary_doubleloop_model import DCMicrogridModel


class Box(object):
    """Small Box space replacement to avoid importing the unmaintained gym package."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self):
        if np.all(np.isfinite(self.low)) and np.all(np.isfinite(self.high)):
            return np.random.uniform(self.low, self.high).astype(self.dtype)
        return np.random.standard_normal(self.shape).astype(self.dtype)

    def __repr__(self):
        low = float(np.min(self.low))
        high = float(np.max(self.high))
        return "Box({}, {}, {}, {})".format(low, high, self.shape, np.dtype(self.dtype).name)


class DCMicrogridEnv(object):
    """
    MAPPO-compatible environment for the HIL-aligned secondary controller.

    Return convention matches the vectorized multi-agent wrapper used by this repository:
    reset() -> obs with shape [num_agents, obs_dim]
    step(actions) -> obs, rewards, dones, infos
    """

    def __init__(self, args):
        self.model = DCMicrogridModel(args)
        self.num_agents = self.model.num_agents
        self.n = self.num_agents
        self.episode_length = self.model.episode_length

        obs_dim = self.model._get_obs().shape[-1]
        share_obs_dim = obs_dim * self.num_agents
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            )
            for _ in range(self.num_agents)
        ]

    def seed(self, seed=None):
        self.model.seed(seed)

    def reset(self):
        return self.model.reset()

    def step(self, actions):
        obs, reward, done, info = self.model.step(actions)
        rewards = np.ones((self.num_agents, 1), dtype=np.float32) * reward
        dones = np.ones(self.num_agents, dtype=np.bool_) * done
        infos = []
        for agent_id in range(self.num_agents):
            agent_info = dict(info)
            agent_info["agent_id"] = agent_id
            agent_info["secondary_voltage_action"] = float(
                self.model.secondary_actions[agent_id]
            )
            agent_info["secondary_voltage_correction"] = float(
                self.model.secondary_actions[agent_id]
            )
            agent_info["bus_voltage"] = float(self.model.bus_voltages[agent_id])
            agent_info["converter_voltage"] = float(
                self.model.converter_voltages[agent_id]
            )
            agent_info["branch_current"] = float(self.model.inductor_currents[agent_id])
            agent_info["battery_current"] = agent_info["branch_current"]
            agent_info["load_current"] = float(self.model.load_currents[agent_id])
            agent_info["load_resistance"] = float(self.model.load_resistances[agent_id])
            agent_info["capacity_ratio"] = float(self.model.capacity_ratios[agent_id])
            agent_info["current_per_capacity"] = float(
                self.model.inductor_currents[agent_id]
                / self.model.capacity_ratios[agent_id]
            )
            agent_info["target_branch_current"] = float(
                info["share_lambda"] * self.model.capacity_ratios[agent_id]
            )
            infos.append(agent_info)
        return obs, rewards, dones, infos

    def close(self):
        pass

    def render(self, mode="human"):
        if mode == "human":
            voltage_error, average_voltage_error, current_error = self.model._errors()
            max_voltage_error = float(
                np.max(np.abs(self.model.bus_voltages - self.model.nominal_voltage))
                / max(1e-6, self.model.nominal_voltage)
            )
            print(
                "t={} avg_voltage={:.3f} local_voltage_error={:.4f} max_voltage_error={:.4f} avg_voltage_error={:.4f} current_error={:.4f} action_change={:.4f}".format(
                    self.model.t,
                    np.mean(self.model.bus_voltages),
                    voltage_error,
                    max_voltage_error,
                    average_voltage_error,
                    current_error,
                    self.model.last_action_change,
                )
            )
            return None
        if mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        raise NotImplementedError

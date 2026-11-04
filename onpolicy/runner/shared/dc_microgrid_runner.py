import time
import numpy as np
import torch

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class DCMicrogridRunner(Runner):#父类Runner，base_runner里定义
    """Runner for the DC microgrid secondary-control environment."""

    def __init__(self, config):
        super(DCMicrogridRunner, self).__init__(config)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):#是否使用学习率逐渐衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_infos = []
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_infos.append(infos)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)#数据存入rollout buffer

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:#打印日志，FPS，每一秒完成的环境步数
                end = time.time()
                print(
                    "\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.env_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / max(end - start, 1e-6)),
                    )
                )

                env_infos = self._collect_env_infos(infos)
                env_infos.update(self._collect_transient_episode_infos(episode_infos))
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print(
                    "dc average voltage {:.3f} V, local voltage error {:.4f}, max voltage error {:.4f}, safe-band ratio {:.3f}, avg voltage error {:.4f}, proportional current error {:.4f}, I/k spread {:.3f} A, action dRMS {:.4f} V, I/k dRMS {:.4f} A, stability penalty {:.4f}, stage count {:.0f}".format(
                        np.mean(env_infos["dc/average_voltage"]),#平均母线电压
                        np.mean(env_infos["dc/voltage_error"]),
                        np.mean(env_infos["dc/max_voltage_error"]),#四条母线中最大的电压偏差
                        np.mean(env_infos["dc/voltage_within_safe_band"]),
                        np.mean(env_infos["dc/average_voltage_error"]),#平均母线电压的误差
                        np.mean(env_infos["dc/current_error"]),
                        np.mean(env_infos["dc/current_per_capacity_spread_a"]),#容量归一化电流最大值与最小值之差
                        np.mean(env_infos["dc/action_delta_rms_v"]),
                        np.mean(env_infos["dc/current_per_capacity_delta_rms_a"]),
                        np.mean(env_infos["dc/weighted_steady_stability_penalty"]),
                        np.mean(env_infos["dc/stage_count"]),
                    )
                )
                print(
                    "dc transient max average deviation {:.3f} V, max bus deviation {:.3f} V, active safe-band ratio {:.3f}, weighted penalty {:.4f}".format(
                        np.mean(env_infos["dc/episode_max_average_voltage_peak_deviation_v"]),
                        np.mean(env_infos["dc/episode_max_bus_voltage_peak_deviation_v"]),
                        np.mean(env_infos["dc/transient_peak_safe_ratio"]),
                        np.mean(env_infos["dc/mean_weighted_transient_voltage_penalty"]),
                    )
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

    def warmup(self):
        obs = self.envs.reset()
        share_obs = self._build_share_obs(obs)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        if self.envs.action_space[0].__class__.__name__ == "Box":
            actions_env = actions.astype(np.float32)
        else:
            raise NotImplementedError("DCMicrogridRunner expects continuous Box actions.")

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = self._build_share_obs(obs)
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    def _build_share_obs(self, obs):
        if self.use_centralized_V:
            share_obs = obs.reshape(obs.shape[0], -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        return share_obs

    def _collect_env_infos(self, infos):
        env_infos = {
            "dc/voltage_error": [],
            "dc/average_voltage_error": [],
            "dc/max_voltage_error": [],
            "dc/voltage_within_safe_band": [],
            "dc/current_error": [],
            "dc/current_per_capacity_spread_a": [],
            "dc/mean_current_per_capacity_error_a": [],
            "dc/action_change": [],
            "dc/action_delta_rms_v": [],
            "dc/current_per_capacity_delta_rms_a": [],
            "dc/steady_action_change_penalty": [],
            "dc/steady_current_ripple_penalty": [],
            "dc/weighted_steady_stability_penalty": [],
            "dc/average_voltage": [],
            "dc/average_current": [],
            "dc/average_load": [],
            "dc/average_pv": [],
            "dc/stage_count": [],
        }
        for env_info in infos:
            first_agent_info = env_info[0]
            env_infos["dc/voltage_error"].append(first_agent_info["voltage_error"])
            env_infos["dc/average_voltage_error"].append(first_agent_info["average_voltage_error"])
            env_infos["dc/max_voltage_error"].append(first_agent_info["max_voltage_error"])
            env_infos["dc/voltage_within_safe_band"].append(first_agent_info["voltage_within_safe_band"])
            env_infos["dc/current_error"].append(first_agent_info["current_error"])
            env_infos["dc/current_per_capacity_spread_a"].append(
                first_agent_info["current_per_capacity_spread_a"]
            )
            env_infos["dc/mean_current_per_capacity_error_a"].append(
                first_agent_info["mean_current_per_capacity_error_a"]
            )
            env_infos["dc/action_change"].append(first_agent_info["action_change"])
            env_infos["dc/action_delta_rms_v"].append(
                first_agent_info["action_delta_rms_v"]
            )
            env_infos["dc/current_per_capacity_delta_rms_a"].append(
                first_agent_info["current_per_capacity_delta_rms_a"]
            )
            env_infos["dc/steady_action_change_penalty"].append(
                first_agent_info["steady_action_change_penalty"]
            )
            env_infos["dc/steady_current_ripple_penalty"].append(
                first_agent_info["steady_current_ripple_penalty"]
            )
            env_infos["dc/weighted_steady_stability_penalty"].append(
                first_agent_info["weighted_steady_stability_penalty"]
            )
            env_infos["dc/average_voltage"].append(first_agent_info["average_voltage"])
            env_infos["dc/average_current"].append(first_agent_info["average_current"])
            env_infos["dc/average_load"].append(first_agent_info["average_load"])
            env_infos["dc/average_pv"].append(first_agent_info["average_pv"])
            env_infos["dc/stage_count"].append(first_agent_info["stage_count"])

        for agent_id in range(self.num_agents):
            env_infos["agent{}/bus_voltage".format(agent_id)] = []
            env_infos["agent{}/battery_current".format(agent_id)] = []
            env_infos["agent{}/secondary_voltage_action".format(agent_id)] = []
            for env_info in infos:
                env_infos["agent{}/bus_voltage".format(agent_id)].append(env_info[agent_id]["bus_voltage"])
                env_infos["agent{}/battery_current".format(agent_id)].append(env_info[agent_id]["battery_current"])
                env_infos["agent{}/secondary_voltage_action".format(agent_id)].append(
                    env_info[agent_id]["secondary_voltage_action"]
                )

        return env_infos

    def _collect_transient_episode_infos(self, episode_infos):
        """Summarise event-window extrema across every step and rollout env."""
        all_infos = [
            env_info[0]
            for step_infos in episode_infos
            for env_info in step_infos
        ]
        transient_infos = [
            info for info in all_infos if info["transient_active"] > 0.5
        ]
        if not transient_infos:
            return {
                "dc/episode_max_average_voltage_peak_deviation_v": [0.0],
                "dc/episode_max_bus_voltage_peak_deviation_v": [0.0],
                "dc/transient_peak_safe_ratio": [1.0],
                "dc/mean_transient_average_voltage_penalty": [0.0],
                "dc/mean_transient_bus_voltage_penalty": [0.0],
                "dc/mean_weighted_transient_voltage_penalty": [0.0],
            }
        return {
            "dc/episode_max_average_voltage_peak_deviation_v": [
                max(info["average_voltage_peak_deviation_v"] for info in transient_infos)
            ],
            "dc/episode_max_bus_voltage_peak_deviation_v": [
                max(info["max_bus_voltage_peak_deviation_v"] for info in transient_infos)
            ],
            "dc/transient_peak_safe_ratio": [
                np.mean(
                    [info["peak_voltage_within_safe_band"] for info in transient_infos]
                )
            ],
            "dc/mean_transient_average_voltage_penalty": [
                np.mean(
                    [
                        info["transient_average_voltage_penalty"]
                        for info in transient_infos
                    ]
                )
            ],
            "dc/mean_transient_bus_voltage_penalty": [
                np.mean(
                    [info["transient_bus_voltage_penalty"] for info in transient_infos]
                )
            ],
            "dc/mean_weighted_transient_voltage_penalty": [
                np.mean(
                    [
                        info["weighted_transient_voltage_penalty"]
                        for info in transient_infos
                    ]
                )
            ],
        }

#!/usr/bin/env python
import sys
import os
import socket
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import wandb
import setproctitle

from onpolicy.config import get_config
from cpl_env import CPLDCMicrogridEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "DCMicrogrid":
                env = CPLDCMicrogridEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--dc_load_model",
        choices=("cpl", "mixed"),
        default="cpl",
        help="Select the historical single-CPL case or unified mixed-load training.",
    )
    parser.add_argument("--dc_cpl_base_power_w", type=float, default=0.0)
    parser.add_argument("--dc_cpl_power_min_w", type=float, default=100.0)
    parser.add_argument("--dc_cpl_power_max_w", type=float, default=300.0)
    parser.add_argument("--dc_cpl_voltage_floor_v", type=float, default=20.0)
    parser.add_argument("--dc_cpl_solver_max_iterations", type=int, default=12)
    parser.add_argument("--dc_cpl_solver_tolerance_a", type=float, default=1.0e-7)
    parser.add_argument("--dc_cpl_low_voltage_v", type=float, default=43.2)
    parser.add_argument("--dc_cpl_current_limit_a", type=float, default=10.0)
    parser.add_argument("--dc_reward_cpl_low_voltage", type=float, default=5.0)
    parser.add_argument("--dc_reward_cpl_overcurrent", type=float, default=2.0)
    parser.add_argument("--dc_reward_cpl_solver_failure", type=float, default=50.0)
    parser.add_argument("--dc_common_voltage_kp", type=float, default=0.5)
    parser.add_argument("--dc_common_voltage_ki", type=float, default=2.0)
    parser.add_argument("--dc_common_voltage_min_v", type=float, default=-1.0)
    parser.add_argument("--dc_common_voltage_max_v", type=float, default=3.0)
    parser.add_argument("--dc_common_action_filter_alpha", type=float, default=0.2)
    parser.add_argument(
        "--dc_common_current_feedforward_gain",
        type=float,
        default=0.0,
        help=(
            "Gain applied to mean(droop_i * IL_i) in the common-voltage path; "
            "zero preserves the historical PI-only controller."
        ),
    )
    parser.add_argument(
        "--dc_common_feedforward_filter_alpha",
        type=float,
        default=0.5,
        help="First-order filter coefficient for the common current feed-forward term.",
    )
    parser.add_argument("--dc_reward_voltage_violation", type=float, default=20.0)
    parser.add_argument("--dc_reward_current_violation", type=float, default=3.0)
    parser.add_argument("--scenario_name", type=str, default="ring_secondary_control")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional result root; runs are saved as <results_dir>/<experiment_name>/runN.",
    )
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--dc_capacity_ratios", type=str, default="1,2,1,1")
    parser.add_argument("--dc_obs_voltage_delta_v", type=float, default=9.6)
    parser.add_argument("--dc_voltage_safe_error", type=float, default=0.05)
    parser.add_argument("--dc_current_safe_error", type=float, default=0.02)
    parser.add_argument("--dc_current_base", type=float, default=10.0)
    parser.add_argument("--dc_load_resistance_min", type=float, default=30.0)
    parser.add_argument("--dc_load_resistance_max", type=float, default=40.0)
    parser.add_argument("--dc_initial_load_resistance_min", type=float, default=40.0)
    parser.add_argument("--dc_initial_load_resistance_max", type=float, default=40.0)
    parser.add_argument("--dc_line_resistance", type=float, default=1.0)
    parser.add_argument("--dc_droop_base_v_per_a", type=float, default=1.0)
    parser.add_argument("--dc_input_voltage", type=float, default=100.0)
    parser.add_argument("--dc_inductance", type=float, default=0.005)
    parser.add_argument("--dc_capacitance", type=float, default=0.0022)
    parser.add_argument("--dc_source_resistances", type=str, default="0.01,0.005,0.008,0.008")
    parser.add_argument("--dc_current_loop_dt", type=float, default=0.0001)
    parser.add_argument("--dc_voltage_loop_dt", type=float, default=0.001)
    parser.add_argument("--dc_secondary_loop_dt", type=float, default=0.010)
    parser.add_argument("--dc_voltage_pi_kp", type=float, default=0.30)
    parser.add_argument("--dc_voltage_pi_ki", type=float, default=25.0)
    parser.add_argument("--dc_current_pi_kp", type=float, default=0.12)
    parser.add_argument("--dc_current_pi_ki", type=float, default=0.25)
    parser.add_argument("--dc_duty_min", type=float, default=0.0)
    parser.add_argument("--dc_duty_max", type=float, default=1.0)
    parser.add_argument("--dc_noise_std", type=float, default=0.005)
    parser.add_argument("--dc_current_noise_std", type=float, default=0.01)
    parser.add_argument(
        "--dc_action_filter_alpha",
        type=float,
        default=1.0,
        help="First-order Actor action filter coefficient; 1.0 preserves legacy behavior.",
    )
    parser.add_argument("--dc_load_random_walk", type=float, default=0.0)
    parser.add_argument("--dc_load_step_time", type=int, default=30)
    parser.add_argument("--dc_load_restore_time", type=int, default=100)
    parser.add_argument("--dc_randomize_load_times", action="store_true", default=False)
    parser.add_argument("--dc_load_step_time_min", type=int, default=20)
    parser.add_argument("--dc_load_step_time_max", type=int, default=60)
    parser.add_argument("--dc_load_step_duration_min", type=int, default=60)
    parser.add_argument("--dc_load_step_duration_max", type=int, default=80)
    parser.add_argument("--dc_load_step_agent", type=int, default=1)
    parser.add_argument("--dc_randomize_load_step_agent", action="store_true", default=False)
    parser.add_argument("--dc_load_step_target_min", type=float, default=30.0)
    parser.add_argument("--dc_load_step_target_max", type=float, default=30.0)
    parser.add_argument("--dc_average_voltage_band_v", type=float, default=0.5)
    parser.add_argument("--dc_settling_steps", type=int, default=20)
    parser.add_argument("--dc_stability_window_steps", type=int, default=50)
    parser.add_argument("--dc_reward_average_voltage", type=float, default=14.0)
    parser.add_argument("--dc_reward_current_share", type=float, default=12.0)
    parser.add_argument("--dc_settling_penalty", type=float, default=12.0)
    parser.add_argument(
        "--dc_reward_action_change",
        type=float,
        default=0.0,
        help="Weight for steady-state squared secondary-action movement.",
    )
    parser.add_argument(
        "--dc_action_change_scale_v",
        type=float,
        default=0.10,
        help="Voltage scale used to normalise the action-change penalty.",
    )
    parser.add_argument(
        "--dc_reward_current_ripple",
        type=float,
        default=0.0,
        help="Weight for steady-state capacity-normalised current movement.",
    )
    parser.add_argument(
        "--dc_current_ripple_scale_a",
        type=float,
        default=0.05,
        help="Current scale used to normalise the temporal ripple penalty.",
    )
    parser.add_argument("--dc_transient_window_steps", type=int, default=10)
    parser.add_argument("--dc_transient_average_voltage_band_v", type=float, default=0.25)
    parser.add_argument("--dc_transient_bus_voltage_band_v", type=float, default=2.4)
    parser.add_argument("--dc_reward_transient_average_voltage", type=float, default=0.0)
    parser.add_argument("--dc_reward_transient_bus_voltage", type=float, default=0.0)
    parser.add_argument("--dc_mixed_resistance_initial_min_ohm", type=float, default=40.0)
    parser.add_argument("--dc_mixed_resistance_initial_max_ohm", type=float, default=80.0)
    parser.add_argument("--dc_mixed_resistance_target_min_ohm", type=float, default=20.0)
    parser.add_argument("--dc_mixed_resistance_target_max_ohm", type=float, default=80.0)
    parser.add_argument("--dc_mixed_cpl_initial_max_w", type=float, default=100.0)
    parser.add_argument("--dc_mixed_cpl_target_max_w", type=float, default=500.0)
    parser.add_argument("--dc_mixed_total_load_power_max_w", type=float, default=900.0)
    parser.add_argument("--dc_mixed_event_bus_count_min", type=int, default=1)
    parser.add_argument("--dc_mixed_event_bus_count_max", type=int, default=4)
    parser.add_argument("--dc_mixed_resistive_probability", type=float, default=0.30)
    parser.add_argument("--dc_mixed_cpl_probability", type=float, default=0.35)
    parser.add_argument("--dc_mixed_combined_probability", type=float, default=0.35)
    parser.add_argument("--dc_mixed_ramp_probability", type=float, default=0.35)
    parser.add_argument("--dc_mixed_ramp_steps_min", type=int, default=2)
    parser.add_argument("--dc_mixed_ramp_steps_max", type=int, default=10)
    parser.add_argument(
        "--dc_mixed_curriculum_episodes",
        type=int,
        default=0,
        help="Episodes per worker used to reach the full envelope; 0 derives it from 500k steps.",
    )
    parser.add_argument(
        "--dc_fast_common_voltage",
        action="store_true",
        default=False,
        help="Run the deterministic common-voltage PI at the 1 kHz voltage-loop rate.",
    )
    parser.add_argument("--dc_fast_common_action_filter_alpha", type=float, default=0.10)
    parser.add_argument("--dc_mixed_average_nadir_v", type=float, default=43.2)
    parser.add_argument("--dc_mixed_average_zenith_v", type=float, default=52.8)
    parser.add_argument("--dc_reward_mixed_average_nadir", type=float, default=20.0)
    parser.add_argument("--dc_reward_mixed_average_zenith", type=float, default=20.0)
    parser.add_argument("--dc_reward_mixed_transient_iae", type=float, default=8.0)
    parser.add_argument("--dc_reward_mixed_duty_saturation", type=float, default=4.0)
    parser.add_argument("--dc_reward_mixed_common_saturation", type=float, default=1.0)
    return parser.parse_args(args)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError("DCMicrogrid currently supports rmappo and mappo.")

    if not all_args.share_policy:
        raise NotImplementedError("DCMicrogrid currently provides a shared-policy runner.")

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if all_args.results_dir:
        run_dir = Path(all_args.results_dir).expanduser().resolve() / all_args.experiment_name
    else:
        run_dir = (
            Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
            / all_args.env_name
            / all_args.scenario_name
            / all_args.algorithm_name
            / all_args.experiment_name
        )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            curr_run = "run1" if len(exst_run_nums) == 0 else "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from onpolicy.runner.shared.dc_microgrid_runner import DCMicrogridRunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

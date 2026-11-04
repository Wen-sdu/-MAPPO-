#!/usr/bin/env python
"""Deterministically evaluate a feed-forward MAPPO actor.

Run without arguments for the graphical workflow:
select actor.pt -> select output directory -> evaluate normal and noise0 cases.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _restart_in_project_environment_if_needed() -> None:
    """Make direct/double-click execution use the training virtual environment."""
    required = ("numpy", "torch")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    candidates = (
        project_root / ".venv_mappo" / "Scripts" / "python.exe",
        project_root / ".venv_mappo" / "bin" / "python",
    )
    current_python = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != current_python:
            completed = subprocess.run(
                [str(candidate), str(script_path), *sys.argv[1:]],
                check=False,
            )
            raise SystemExit(completed.returncode)

    missing = [
        name for name in required if importlib.util.find_spec(name) is None
    ]
    raise ModuleNotFoundError(
        "Missing evaluation dependencies "
        f"{missing}; expected project environment at {candidates[0]}"
    )


_restart_in_project_environment_if_needed()

from tkinter import Tk, filedialog, messagebox

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = TRAINING_ROOT / "results"
TRAINING_CODE_ROOT = TRAINING_ROOT / "training"
DEFAULT_STEPS = 2000
DEFAULT_SEED = 1
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_ROOT))

from cpl_env import CPLDCMicrogridEnv
from train_dc_microgrid_cpl import parse_args as parse_training_args
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.config import get_config
from onpolicy.envs.dc_microgrid.DC_Microgrid_Env import DCMicrogridEnv


def find_run_config(actor_path: Path) -> Path:
    run_dir = actor_path.parent.parent
    for candidate in (run_dir / "config.json", run_dir.parent / "launch_config.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No config.json was found beside {actor_path}")


def load_saved_arguments(actor_path: Path) -> list[str]:
    content = json.loads(find_run_config(actor_path).read_text(encoding="utf-8"))
    arguments = content.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError("Saved config has no valid arguments list")
    return arguments


def load_state_dict(actor_path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(actor_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(actor_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(key.startswith("rnn.") for key in state):
        raise ValueError("Selected checkpoint contains an RNN and is not a MAPPO feed-forward actor")
    return state


def unique_output_path(path: Path) -> Path:
    """Preserve an earlier evaluation instead of silently overwriting it."""
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def build_actor(actor_path: Path, env):
    saved = load_saved_arguments(actor_path)
    args = parse_training_args(saved, get_config())
    args.algorithm_name = "mappo"
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    state = load_state_dict(actor_path)
    args.hidden_size = int(state["base.mlp.fc1.0.weight"].shape[0])
    actor = R_Actor(args, env.observation_space[0], env.action_space[0])
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor, args


def _settling_time_s(
    rows: list[dict],
    start_index: int,
    end_index: int,
    band_v: float,
    hold_steps: int = 20,
) -> float | None:
    """Return the first band-entry time that remains valid for hold_steps."""
    if end_index - start_index < hold_steps:
        return None
    event_time_s = 0.0 if start_index == 0 else float(rows[start_index]["time_s"])
    for index in range(start_index, end_index - hold_steps + 1):
        window = rows[index : index + hold_steps]
        if all(abs(float(row["average_voltage"]) - 48.0) <= band_v for row in window):
            return float(rows[index]["time_s"]) - event_time_s
    return None


def _stage_metrics(
    rows: list[dict], start_index: int, end_index: int
) -> dict[str, float | None]:
    stage = rows[start_index:end_index]
    voltage = np.asarray([row["average_voltage"] for row in stage], dtype=float)
    current_error = np.asarray([row["current_error"] for row in stage], dtype=float)
    return {
        "minimum_average_voltage_v": float(np.min(voltage)),
        "maximum_average_voltage_v": float(np.max(voltage)),
        "peak_average_voltage_deviation_v": float(np.max(np.abs(voltage - 48.0))),
        "mean_current_error": float(np.mean(current_error)),
        "settling_time_0p25_v_s": _settling_time_s(
            rows, start_index, end_index, 0.25
        ),
        "settling_time_0p05_v_s": _settling_time_s(
            rows, start_index, end_index, 0.05
        ),
    }


def _transient_summary(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    event_indices = [
        index for index, row in enumerate(rows) if float(row["load_event"]) > 0.5
    ]
    result: dict[str, dict[str, float | None]] = {}
    first_event = event_indices[0] if event_indices else len(rows)
    result["startup"] = _stage_metrics(rows, 0, first_event)
    for event_number, start_index in enumerate(event_indices):
        end_index = (
            event_indices[event_number + 1]
            if event_number + 1 < len(event_indices)
            else len(rows)
        )
        previous_power = (
            float(rows[start_index - 1]["total_cpl_power_w"])
            if start_index > 0
            else float(rows[start_index]["cpl_base_power_w"])
        )
        active_power = float(rows[start_index]["total_cpl_power_w"])
        label = "cpl_add" if active_power > previous_power else "cpl_remove"
        result[label] = {
            "event_time_s": float(rows[start_index]["time_s"]),
            "power_before_w": previous_power,
            "power_after_w": active_power,
            **_stage_metrics(rows, start_index, end_index),
        }
    return result


def evaluate(
    actor_path: Path,
    output_csv: Path,
    steps: int,
    seed: int,
    noise0: bool,
    controller_overrides: dict[str, float] | None = None,
) -> dict:
    saved = load_saved_arguments(actor_path)
    args = parse_training_args(saved, get_config())
    args.algorithm_name = "mappo"
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.episode_length = steps
    args.dc_randomize_load_times = False
    args.dc_randomize_load_step_agent = False
    args.dc_load_step_time = min(200, max(1, steps // 4))
    args.dc_load_restore_time = min(600, max(args.dc_load_step_time + 1, 3 * steps // 4))
    if getattr(args, "dc_load_model", "cpl") == "mixed":
        # Evaluation must exercise the full requested envelope rather than the
        # first, deliberately mild curriculum stage.
        args.dc_mixed_curriculum_episodes = 1
    if noise0:
        args.dc_noise_std = 0.0
        args.dc_current_noise_std = 0.0
    if controller_overrides:
        allowed = {
            "dc_common_voltage_kp",
            "dc_common_voltage_ki",
            "dc_common_voltage_min_v",
            "dc_common_voltage_max_v",
            "dc_common_action_filter_alpha",
            "dc_fast_common_action_filter_alpha",
            "dc_common_current_feedforward_gain",
            "dc_common_feedforward_filter_alpha",
        }
        unknown = set(controller_overrides) - allowed
        if unknown:
            raise ValueError(f"Unsupported controller overrides: {sorted(unknown)}")
        for name, value in controller_overrides.items():
            setattr(args, name, float(value))
            if (
                name == "dc_common_action_filter_alpha"
                and getattr(args, "dc_load_model", "cpl") == "mixed"
            ):
                args.dc_fast_common_action_filter_alpha = float(value)

    env_class = (
        CPLDCMicrogridEnv
        if "--dc_cpl_power_min_w" in saved
        else DCMicrogridEnv
    )
    env = env_class(args)
    env.seed(seed)
    obs = env.reset()
    actor, args = build_actor(actor_path, env)
    rnn = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), np.float32)
    masks = np.ones((args.num_agents, 1), np.float32)
    rows: list[dict] = []

    for step in range(1, steps + 1):
        with torch.no_grad():
            action_t, _, _ = actor(obs, rnn, masks, deterministic=True)
        action = action_t.cpu().numpy()
        obs, rewards, dones, infos = env.step(action)
        info = infos[0]
        row = {
            "step": step,
            "time_s": step * env.model.secondary_dt,
            "reward": float(rewards[0, 0]),
            "average_voltage": info["average_voltage"],
            "average_voltage_error": info["average_voltage_error"],
            "max_voltage_error": info["max_voltage_error"],
            "current_error": info["current_error"],
            "action_change": info["action_change"],
            "action_delta_rms_v": info["action_delta_rms_v"],
            "current_per_capacity_delta_rms_a": info[
                "current_per_capacity_delta_rms_a"
            ],
            "weighted_steady_stability_penalty": info[
                "weighted_steady_stability_penalty"
            ],
            "load_event": info["load_event"],
            "cpl_base_power_w": info.get("cpl_base_power_w", 0.0),
            "active_cpl_power_w": info.get("active_cpl_power_w", 0.0),
            "total_cpl_power_w": info.get("total_cpl_power_w", 0.0),
            "cpl_step_agent": info.get("cpl_step_agent", -1),
            "cpl_minimum_bus_voltage_v": info.get(
                "cpl_minimum_bus_voltage_v", np.min(env.model.bus_voltages)
            ),
            "cpl_maximum_inductor_current_a": info.get(
                "cpl_maximum_inductor_current_a",
                np.max(np.abs(env.model.inductor_currents)),
            ),
            "cpl_solver_converged": info.get("cpl_solver_converged", 1.0),
            "weighted_cpl_safety_penalty": info.get(
                "weighted_cpl_safety_penalty", 0.0
            ),
            "common_voltage_error_v": info.get("common_voltage_error_v", 0.0),
            "common_voltage_integral_v": info.get(
                "common_voltage_integral_v", 0.0
            ),
            "common_voltage_action_v": info.get(
                "common_voltage_action_v", 0.0
            ),
            "common_voltage_filtered_v": info.get(
                "common_voltage_filtered_v", 0.0
            ),
            "common_feedforward_target_v": info.get(
                "common_feedforward_target_v", 0.0
            ),
            "common_feedforward_filtered_v": info.get(
                "common_feedforward_filtered_v", 0.0
            ),
            "mixed_scenario_kind": info.get("mixed_scenario_kind", "legacy"),
            "mixed_event_bus_count": info.get("mixed_event_bus_count", 0),
            "mixed_ramp_steps": info.get("mixed_ramp_steps", 1),
            "total_load_power_w": info.get("total_load_power_w", 0.0),
            "mixed_minimum_average_voltage_v": info.get(
                "mixed_minimum_average_voltage_v", 0.0
            ),
            "mixed_maximum_average_voltage_v": info.get(
                "mixed_maximum_average_voltage_v", 0.0
            ),
            "mixed_average_nadir_penalty": info.get(
                "mixed_average_nadir_penalty", 0.0
            ),
            "mixed_average_zenith_penalty": info.get(
                "mixed_average_zenith_penalty", 0.0
            ),
            "actor_action_mean_v": info.get("actor_action_mean_v", 0.0),
            "voltage_violation": info.get("voltage_violation", 0.0),
            "current_violation": info.get("current_violation", 0.0),
            "weighted_constraint_penalty": info.get(
                "weighted_constraint_penalty", 0.0
            ),
        }
        for agent in range(args.num_agents):
            row[f"raw_u{agent}"] = float(action[agent, 0])
            row[f"du{agent}"] = float(
                info.get(
                    f"actor_differential_action_{agent}_v",
                    action[agent, 0] - np.mean(action[:, 0]),
                )
            )
            row[f"duf{agent}"] = float(
                info.get(
                    f"actor_differential_filtered_{agent}_v",
                    row[f"du{agent}"],
                )
            )
            row[f"u{agent}"] = float(env.model.secondary_actions[agent])
            row[f"V{agent}"] = float(env.model.bus_voltages[agent])
            row[f"I{agent}"] = float(env.model.inductor_currents[agent])
            row[f"cpl_power_{agent}_w"] = float(
                getattr(env.model, "cpl_powers_w", np.zeros(args.num_agents))[agent]
            )
            row[f"initial_R{agent + 1}_ohm"] = info.get(
                f"initial_R{agent + 1}_ohm", 0.0
            )
            row[f"target_R{agent + 1}_ohm"] = info.get(
                f"target_R{agent + 1}_ohm", 0.0
            )
            row[f"initial_P{agent + 1}_w"] = info.get(
                f"initial_P{agent + 1}_w", 0.0
            )
            row[f"target_P{agent + 1}_w"] = info.get(
                f"target_P{agent + 1}_w", 0.0
            )
            row[f"event_bus_{agent + 1}"] = info.get(
                f"event_bus_{agent + 1}", 0.0
            )
        rows.append(row)
        if np.all(dones) and step < steps:
            raise RuntimeError(f"Environment terminated early at step {step}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    average_voltage = np.asarray([row["average_voltage"] for row in rows])
    current_error = np.asarray([row["current_error"] for row in rows])
    action_change = np.asarray([row["action_change"] for row in rows])
    action_delta_rms = np.asarray([row["action_delta_rms_v"] for row in rows])
    common_voltage_action = np.asarray(
        [row["common_voltage_action_v"] for row in rows]
    )
    current_delta_rms = np.asarray(
        [row["current_per_capacity_delta_rms_a"] for row in rows]
    )
    minimum_bus_voltage = min(
        min(row[f"V{agent}"] for agent in range(args.num_agents))
        for row in rows
    )
    maximum_inductor_current = max(
        max(abs(row[f"I{agent}"]) for agent in range(args.num_agents))
        for row in rows
    )
    summary = {
        "actor": str(actor_path),
        "steps": steps,
        "seed": seed,
        "noise0": noise0,
        "common_voltage_kp": float(env.model.common_voltage_kp),
        "common_voltage_ki": float(env.model.common_voltage_ki),
        "common_voltage_min_v": float(env.model.common_voltage_min_v),
        "common_voltage_max_v": float(env.model.common_voltage_max_v),
        "common_action_filter_alpha": float(
            env.model.fast_common_action_filter_alpha
            if getattr(env.model, "fast_common_voltage_enabled", False)
            else env.model.common_action_filter_alpha
        ),
        "common_current_feedforward_gain": float(
            getattr(env.model, "common_current_feedforward_gain", 0.0)
        ),
        "common_feedforward_filter_alpha": float(
            getattr(env.model, "common_feedforward_filter_alpha", 0.0)
        ),
        "mixed_average_nadir_v": float(
            getattr(env.model, "mixed_average_nadir_v", 0.0)
        ),
        "mixed_average_zenith_v": float(
            getattr(env.model, "mixed_average_zenith_v", 0.0)
        ),
        "mixed_scenario_kind": str(rows[0]["mixed_scenario_kind"]),
        "mixed_event_bus_count": int(rows[0]["mixed_event_bus_count"]),
        "mixed_ramp_steps": int(rows[0]["mixed_ramp_steps"]),
        "min_average_voltage": float(np.min(average_voltage)),
        "max_average_voltage": float(np.max(average_voltage)),
        "mean_current_error": float(np.mean(current_error)),
        "mean_action_change": float(np.mean(action_change)),
        "mean_action_delta_rms_v": float(np.mean(action_delta_rms)),
        "final_common_voltage_action_v": float(common_voltage_action[-1]),
        "min_common_voltage_action_v": float(np.min(common_voltage_action)),
        "max_common_voltage_action_v": float(np.max(common_voltage_action)),
        "mean_current_per_capacity_delta_rms_a": float(
            np.mean(current_delta_rms)
        ),
        "minimum_bus_voltage_v": float(minimum_bus_voltage),
        "maximum_inductor_current_a": float(maximum_inductor_current),
        "cpl_solver_all_converged": bool(
            all(row["cpl_solver_converged"] > 0.5 for row in rows)
        ),
        "baseline_cpl_power_w": float(
            max(row["cpl_base_power_w"] for row in rows)
        ),
        "active_cpl_power_w": float(
            max(row["total_cpl_power_w"] for row in rows)
        ),
        "cpl_step_agent": int(
            max(row["cpl_step_agent"] for row in rows)
        ),
        "stages": _transient_summary(rows),
    }
    output_csv.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a feed-forward MAPPO actor")
    parser.add_argument("actor", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise0", action="store_true")
    parser.add_argument("--common-kp", type=float, default=None)
    parser.add_argument("--common-ki", type=float, default=None)
    parser.add_argument("--common-alpha", type=float, default=None)
    parser.add_argument("--common-min", type=float, default=None)
    parser.add_argument("--common-max", type=float, default=None)
    parser.add_argument("--common-ff-gain", type=float, default=None)
    parser.add_argument("--common-ff-alpha", type=float, default=None)
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> int:
    args = cli(argv)
    actor = args.actor.resolve()
    if not actor.is_file():
        raise FileNotFoundError(actor)
    output = args.output
    if output is None:
        output = actor.parent.parent / "evaluation" / (
            "mappo_noise0.csv" if args.noise0 else "mappo_normal.csv"
        )
    overrides = {
        name: value
        for name, value in {
            "dc_common_voltage_kp": args.common_kp,
            "dc_common_voltage_ki": args.common_ki,
            "dc_common_action_filter_alpha": args.common_alpha,
            "dc_common_voltage_min_v": args.common_min,
            "dc_common_voltage_max_v": args.common_max,
            "dc_common_current_feedforward_gain": args.common_ff_gain,
            "dc_common_feedforward_filter_alpha": args.common_ff_alpha,
        }.items()
        if value is not None
    }
    summary = evaluate(
        actor,
        output.resolve(),
        args.steps,
        args.seed,
        args.noise0,
        overrides,
    )
    print(json.dumps(summary, indent=2))
    print(f"CSV: {output.resolve()}")
    return 0


def select_actor_file(root: Tk) -> Path | None:
    file_path = filedialog.askopenfilename(
        parent=root,
        title="选择 MAPPO 训练生成的 actor.pt",
        initialdir=str(RESULTS_ROOT if RESULTS_ROOT.exists() else TRAINING_ROOT),
        filetypes=(("Actor checkpoint", "actor.pt"), ("PyTorch checkpoint", "*.pt")),
    )
    return Path(file_path).resolve() if file_path else None


def select_output_directory(root: Tk, actor_path: Path) -> Path | None:
    default_dir = actor_path.parent.parent / "evaluation"
    default_dir.mkdir(parents=True, exist_ok=True)
    folder = filedialog.askdirectory(
        parent=root,
        title="选择评估结果保存文件夹",
        initialdir=str(default_dir),
        mustexist=False,
    )
    return Path(folder).resolve() if folder else None


def gui_main() -> int:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        actor_path = select_actor_file(root)
        if actor_path is None:
            return 0
        if not actor_path.is_file():
            raise FileNotFoundError(f"Actor checkpoint does not exist: {actor_path}")

        output_dir = select_output_directory(root, actor_path)
        if output_dir is None:
            return 0

        generated: list[Path] = []
        summaries: list[str] = []
        for mode, noise0 in (("normal", False), ("noise0", True)):
            output_csv = unique_output_path(
                output_dir / f"{mode}_seed{DEFAULT_SEED}.csv"
            )
            summary = evaluate(
                actor_path,
                output_csv,
                DEFAULT_STEPS,
                DEFAULT_SEED,
                noise0,
            )
            generated.append(output_csv)
            summaries.append(
                f"{mode}: Vavg={summary['min_average_voltage']:.3f}~"
                f"{summary['max_average_voltage']:.3f} V, "
                f"mean current error={summary['mean_current_error']:.4f}, "
                f"CPL={summary['baseline_cpl_power_w']:.1f}->"
                f"{summary['active_cpl_power_w']:.1f} W, "
                f"Vmin={summary['minimum_bus_voltage_v']:.3f} V, "
                f"Imax={summary['maximum_inductor_current_a']:.3f} A, "
                f"solver={'OK' if summary['cpl_solver_all_converged'] else 'FAILED'}"
            )

        messagebox.showinfo(
            "MAPPO评估完成",
            "已生成两个CSV文件：\n"
            + "\n".join(str(path) for path in generated)
            + "\n\n结果摘要：\n"
            + "\n".join(summaries),
            parent=root,
        )
        return 0
    except Exception as exc:
        messagebox.showerror("MAPPO评估失败", str(exc), parent=root)
        return 1
    finally:
        root.destroy()


def main() -> int:
    return cli_main(sys.argv[1:]) if len(sys.argv) > 1 else gui_main()


if __name__ == "__main__":
    raise SystemExit(main())


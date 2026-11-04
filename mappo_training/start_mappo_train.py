#!/usr/bin/env python
"""Launch feed-forward MAPPO training for the DC microgrid."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
DEFAULT_CONFIG = TRAINING_ROOT / "configs" / "final_mixed_train_config.json"
TRAIN_SCRIPT = TRAINING_ROOT / "training" / "train_dc_microgrid_cpl.py"
RESULTS_ROOT = TRAINING_ROOT / "results"


def usable_python(candidate: Path | str) -> Path | None:
    path = Path(candidate)
    if not path.is_file():
        return None
    probe = subprocess.run(
        [str(path), "-c", "import torch, numpy"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return path.resolve() if probe.returncode == 0 else None


def project_python() -> Path:
    candidates: list[Path | str] = [
        PROJECT_ROOT / ".venv_mappo" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv_hil32" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    system_python = shutil.which("python")
    if system_python:
        candidates.append(system_python)
    for candidate in candidates:
        result = usable_python(candidate)
        if result is not None:
            return result
    raise FileNotFoundError(
        "No Python with torch and numpy was found. Create .venv_mappo and "
        "install requirements_dc_microgrid.txt."
    )


def load_config(path: Path) -> list[str]:
    content = json.loads(path.read_text(encoding="utf-8"))
    arguments = content.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError(f"Invalid arguments list in {path}")
    if "--algorithm_name" not in arguments:
        raise ValueError("Config must explicitly select --algorithm_name mappo")
    index = arguments.index("--algorithm_name")
    if index + 1 >= len(arguments) or arguments[index + 1] != "mappo":
        raise ValueError("This launcher accepts feed-forward MAPPO only")
    return list(arguments)


def resolve_project_path(path: Path | str) -> Path:
    """Resolve package paths independently of the caller's working directory."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def replace_option(arguments: list[str], name: str, value: str) -> None:
    if name in arguments:
        index = arguments.index(name)
        if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
            raise ValueError(f"{name} does not have a value")
        arguments[index + 1] = value
    else:
        arguments.extend([name, value])


def run_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("run"))
    except ValueError:
        return -1


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feed-forward MAPPO training launcher")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing actor.pt and critic.pt for weight warm start.",
    )
    parser.add_argument("--no-pause", action="store_true")
    return parser.parse_args()


def main() -> int:
    launcher = parse_args()
    config_path = resolve_project_path(launcher.config)
    arguments = load_config(config_path)

    if launcher.steps is not None:
        if launcher.steps <= 0:
            raise ValueError("--steps must be positive")
        replace_option(arguments, "--num_env_steps", str(launcher.steps))

    warm_start_source = launcher.model_dir
    if warm_start_source is None and "--model_dir" in arguments:
        model_index = arguments.index("--model_dir")
        if (
            model_index + 1 >= len(arguments)
            or arguments[model_index + 1].startswith("--")
        ):
            raise ValueError("--model_dir does not have a value")
        warm_start_source = Path(arguments[model_index + 1])

    warm_start_dir = None
    if warm_start_source is not None:
        warm_start_dir = resolve_project_path(warm_start_source)
        for checkpoint_name in ("actor.pt", "critic.pt"):
            checkpoint = warm_start_dir / checkpoint_name
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Warm-start checkpoint was not found: {checkpoint}"
                )
        replace_option(arguments, "--model_dir", str(warm_start_dir))

    hidden_index = arguments.index("--hidden_size")
    hidden_size = int(arguments[hidden_index + 1])
    steps_index = arguments.index("--num_env_steps")
    num_env_steps = int(arguments[steps_index + 1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    load_model = "mixed"
    if "--dc_load_model" in arguments:
        load_model = arguments[arguments.index("--dc_load_model") + 1]
    experiment = launcher.experiment or (
        f"mappo_{load_model}_ff{hidden_size}_{num_env_steps}_{timestamp}"
    )

    replace_option(arguments, "--experiment_name", experiment)
    replace_option(arguments, "--results_dir", str(RESULTS_ROOT.resolve()))

    python = project_python()
    command = [str(python), "-u", str(TRAIN_SCRIPT), *arguments]
    experiment_root = RESULTS_ROOT / experiment
    existing_runs = {
        p.resolve() for p in experiment_root.glob("run*")
        if p.is_dir() and run_number(p) >= 0
    }

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "algorithm": "mappo",
        "recurrent": False,
        "load_model": load_model,
        "warm_start_model_dir": str(warm_start_dir) if warm_start_dir else None,
        "experiment": experiment,
        "project_root": str(PROJECT_ROOT),
        "python": str(python),
        "training_script": str(TRAIN_SCRIPT),
        "source_config": str(config_path),
        "arguments": arguments,
        "command": command,
    }
    write_json(experiment_root / "launch_config.json", metadata)

    print("=" * 72)
    print("DC microgrid feed-forward MAPPO mixed-load training")
    print(f"Load model : {load_model}")
    print(f"Experiment : {experiment}")
    print(f"Steps      : {num_env_steps:,}")
    print(f"Hidden     : {hidden_size}")
    print(f"Warm start : {warm_start_dir if warm_start_dir else 'none'}")
    print(f"Results    : {experiment_root}")
    print("=" * 72)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )

    runs = sorted(
        (p for p in experiment_root.glob("run*") if p.is_dir() and run_number(p) >= 0),
        key=run_number,
    )
    new_runs = [p for p in runs if p.resolve() not in existing_runs]
    run_dir = new_runs[-1] if new_runs else (runs[-1] if runs else None)

    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata["exit_code"] = completed.returncode
    metadata["status"] = "completed" if completed.returncode == 0 else "failed"
    metadata["run_dir"] = str(run_dir) if run_dir else None
    write_json(experiment_root / "launch_config.json", metadata)
    if run_dir is not None:
        write_json(run_dir / "config.json", metadata)
        (run_dir / "evaluation").mkdir(exist_ok=True)

    if completed.returncode != 0:
        print(f"Training failed with exit code {completed.returncode}.")
        return completed.returncode
    if run_dir is None:
        print("Training completed but no run directory was found.")
        return 1

    print("Training completed successfully.")
    print(f"Actor : {run_dir / 'models' / 'actor.pt'}")
    print(f"Critic: {run_dir / 'models' / 'critic.pt'}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except KeyboardInterrupt:
        print("Training interrupted.")
        code = 130
    except Exception as exc:
        print(f"Unable to start training: {exc}")
        code = 1
    finally:
        help_requested = "--help" in sys.argv or "-h" in sys.argv
        if "--no-pause" not in sys.argv and not help_requested and sys.stdin.isatty():
            try:
                input("Press Enter to close...")
            except EOFError:
                pass
    raise SystemExit(code)


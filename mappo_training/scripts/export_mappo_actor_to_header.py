#!/usr/bin/env python
"""Export a feed-forward MAPPO actor to a C header."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import numpy as np
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
RESULTS_ROOT = TRAINING_ROOT / "results"
DEFAULT_OUTPUT = PROJECT_ROOT / "ccs_deployment" / "source" / "mappo_actor_weights.h"
CAPACITY_RATIOS = np.asarray([1.0, 2.0, 1.0, 1.0], dtype=np.float32)
EXPECTED_OBS_DIM = 6
SUPPORTED_HIDDEN_DIMS = (32, 64)

TENSOR_MAP = (
    ("MAPPO_FEATURE_NORM_WEIGHT", "base.feature_norm.weight"),
    ("MAPPO_FEATURE_NORM_BIAS", "base.feature_norm.bias"),
    ("MAPPO_FC1_WEIGHT", "base.mlp.fc1.0.weight"),
    ("MAPPO_FC1_BIAS", "base.mlp.fc1.0.bias"),
    ("MAPPO_FC1_NORM_WEIGHT", "base.mlp.fc1.2.weight"),
    ("MAPPO_FC1_NORM_BIAS", "base.mlp.fc1.2.bias"),
    ("MAPPO_FC2_WEIGHT", "base.mlp.fc2.0.0.weight"),
    ("MAPPO_FC2_BIAS", "base.mlp.fc2.0.0.bias"),
    ("MAPPO_FC2_NORM_WEIGHT", "base.mlp.fc2.0.2.weight"),
    ("MAPPO_FC2_NORM_BIAS", "base.mlp.fc2.0.2.bias"),
    ("MAPPO_ACTION_WEIGHT", "act.action_out.fc_mean.weight"),
    ("MAPPO_ACTION_BIAS", "act.action_out.fc_mean.bias"),
)


def load_actor_state_dict(actor_path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(actor_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(actor_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def validate_state_dict(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    if any(key.startswith("rnn.") for key in state):
        raise ValueError("Checkpoint contains rnn.* parameters; export a MAPPO model, not RMAPPO")
    required = {torch_name for _, torch_name in TENSOR_MAP}
    missing = required - set(state)
    if missing:
        raise KeyError("Missing Actor parameters: " + ", ".join(sorted(missing)))
    allowed = required | {"act.action_out.logstd._bias"}
    unexpected = set(state) - allowed
    if unexpected:
        raise ValueError("Unsupported Actor parameters: " + ", ".join(sorted(unexpected)))

    obs_dim = int(state["base.feature_norm.weight"].numel())
    hidden_dim = int(state["base.mlp.fc1.0.weight"].shape[0])
    if obs_dim != EXPECTED_OBS_DIM:
        raise ValueError(f"Expected obs_dim={EXPECTED_OBS_DIM}, got {obs_dim}")
    if hidden_dim not in SUPPORTED_HIDDEN_DIMS:
        raise ValueError(f"Supported hidden dimensions are {SUPPORTED_HIDDEN_DIMS}, got {hidden_dim}")

    shapes = {
        "base.feature_norm.weight": (obs_dim,),
        "base.feature_norm.bias": (obs_dim,),
        "base.mlp.fc1.0.weight": (hidden_dim, obs_dim),
        "base.mlp.fc1.0.bias": (hidden_dim,),
        "base.mlp.fc1.2.weight": (hidden_dim,),
        "base.mlp.fc1.2.bias": (hidden_dim,),
        "base.mlp.fc2.0.0.weight": (hidden_dim, hidden_dim),
        "base.mlp.fc2.0.0.bias": (hidden_dim,),
        "base.mlp.fc2.0.2.weight": (hidden_dim,),
        "base.mlp.fc2.0.2.bias": (hidden_dim,),
        "act.action_out.fc_mean.weight": (1, hidden_dim),
        "act.action_out.fc_mean.bias": (1,),
    }
    for key, expected in shapes.items():
        actual = tuple(state[key].shape)
        if actual != expected:
            raise ValueError(f"{key}: expected {expected}, got {actual}")
    return obs_dim, hidden_dim


def c_float(value: float) -> str:
    value = float(np.float32(value))
    if not np.isfinite(value):
        raise ValueError("Actor contains NaN or infinity")
    if value == 0.0:
        value = 0.0
    return f"{value:.9e}f"


def c_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, flat.size, 6):
        lines.append("    " + ", ".join(c_float(x) for x in flat[start:start + 6]))
    return f"static const float {name}[{flat.size}U] = {{\n" + ",\n".join(lines) + "\n};\n"


def build_header(state: dict[str, torch.Tensor], sha256: str, obs_dim: int, hidden_dim: int) -> str:
    parts = [
        "#ifndef MAPPO_ACTOR_WEIGHTS_H",
        "#define MAPPO_ACTOR_WEIGHTS_H",
        "",
        "/* Generated from a feed-forward deterministic MAPPO Actor. */",
        f'#define MAPPO_ACTOR_SHA256 "{sha256}"',
        f"#define MAPPO_OBS_DIM {obs_dim}U",
        f"#define MAPPO_HIDDEN_DIM {hidden_dim}U",
        "#define MAPPO_USE_GRU 0U",
        "#define MAPPO_LAYER_NORM_EPS 1.0e-5f",
        "#define MAPPO_TRAIN_CAPACITY_RATIO_COUNT 4U",
        c_array("MAPPO_TRAIN_CAPACITY_RATIOS", CAPACITY_RATIOS),
    ]
    for c_name, torch_name in TENSOR_MAP:
        values = state[torch_name].detach().cpu().to(torch.float32).numpy()
        parts.append(c_array(c_name, values))
    parts.extend(["#endif /* MAPPO_ACTOR_WEIGHTS_H */", ""])
    return "\n".join(parts)


def export_actor(actor: Path, output: Path) -> dict[str, object]:
    actor = actor.resolve()
    output = output.resolve()
    if not actor.is_file():
        raise FileNotFoundError(actor)
    state = load_actor_state_dict(actor)
    obs_dim, hidden_dim = validate_state_dict(state)
    sha256 = hashlib.sha256(actor.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_header(state, sha256, obs_dim, hidden_dim),
        encoding="ascii",
        newline="\n",
    )
    count = sum(int(state[key].numel()) for _, key in TENSOR_MAP)
    return {
        "actor": actor,
        "sha256": sha256,
        "output": output,
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "parameter_count": count,
    }


def print_summary(summary: dict[str, object]) -> None:
    print(f"Actor      : {summary['actor']}")
    print(f"SHA256     : {summary['sha256']}")
    print(f"Output     : {summary['output']}")
    print(f"Observation: {summary['obs_dim']}")
    print(f"Hidden     : {summary['hidden_dim']}")
    print(f"Parameters : {summary['parameter_count']}")


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a feed-forward MAPPO actor")
    parser.add_argument("actor", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print_summary(export_actor(args.actor, args.output))
    return 0


def select_actor_file(root: Tk) -> Path | None:
    file_path = filedialog.askopenfilename(
        parent=root,
        title="选择需要导出的 MAPPO actor.pt",
        initialdir=str(RESULTS_ROOT if RESULTS_ROOT.exists() else TRAINING_ROOT),
        filetypes=(("Actor checkpoint", "actor.pt"), ("PyTorch checkpoint", "*.pt")),
    )
    return Path(file_path).resolve() if file_path else None


def select_output_file(root: Tk) -> Path | None:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    file_path = filedialog.asksaveasfilename(
        parent=root,
        title="选择 MAPPO 权重头文件保存位置",
        initialdir=str(DEFAULT_OUTPUT.parent),
        initialfile=DEFAULT_OUTPUT.name,
        defaultextension=".h",
        filetypes=(("C header", "*.h"), ("All files", "*.*")),
        confirmoverwrite=True,
    )
    return Path(file_path).resolve() if file_path else None


def gui_main() -> int:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        actor = select_actor_file(root)
        if actor is None:
            return 0
        output = select_output_file(root)
        if output is None:
            return 0
        summary = export_actor(actor, output)
        print_summary(summary)
        messagebox.showinfo(
            "MAPPO导出完成",
            f"头文件已生成：\n{summary['output']}\n\n"
            f"输入维度：{summary['obs_dim']}\n"
            f"隐藏层维度：{summary['hidden_dim']}\n"
            f"参数数量：{summary['parameter_count']}",
            parent=root,
        )
        return 0
    except Exception as exc:
        messagebox.showerror("MAPPO导出失败", str(exc), parent=root)
        return 1
    finally:
        root.destroy()


def main() -> int:
    return cli_main(sys.argv[1:]) if len(sys.argv) > 1 else gui_main()


if __name__ == "__main__":
    raise SystemExit(main())


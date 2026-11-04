#!/usr/bin/env python
"""Compare PyTorch MAPPO output with the exported feed-forward C data path."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TRAINING_ROOT = SCRIPT_DIR.parent
TRAINING_CODE_ROOT = TRAINING_ROOT / "training"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TRAINING_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_ROOT))

from cpl_env import CPLDCMicrogridEnv
from train_dc_microgrid_cpl import parse_args as parse_training_args
from eval_mappo_deterministic import build_actor, load_saved_arguments
from export_mappo_actor_to_header import TENSOR_MAP
from onpolicy.config import get_config
from onpolicy.envs.dc_microgrid.DC_Microgrid_Env import DCMicrogridEnv


FLOAT_PATTERN = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def parse_header(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text(encoding="ascii")
    arrays: dict[str, np.ndarray] = {}
    for c_name, _ in TENSOR_MAP:
        match = re.search(
            rf"static\s+const\s+float\s+{re.escape(c_name)}\s*\[[^]]+\]\s*=\s*\{{(.*?)\}};",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError(f"Array {c_name} was not found in {path}")
        values = re.findall(FLOAT_PATTERN, match.group(1).replace("f", ""))
        arrays[c_name] = np.asarray([float(value) for value in values], dtype=np.float32)
    return arrays


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True, dtype=np.float32)
    return ((x - mean) / np.sqrt(variance + np.float32(1.0e-5))) * weight + bias


def exported_forward(obs: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    obs_dim = obs.shape[1]
    hidden = arrays["MAPPO_FC1_BIAS"].size
    x = layer_norm(
        obs,
        arrays["MAPPO_FEATURE_NORM_WEIGHT"],
        arrays["MAPPO_FEATURE_NORM_BIAS"],
    )
    w1 = arrays["MAPPO_FC1_WEIGHT"].reshape(hidden, obs_dim)
    x = np.maximum(x @ w1.T + arrays["MAPPO_FC1_BIAS"], 0.0)
    x = layer_norm(x, arrays["MAPPO_FC1_NORM_WEIGHT"], arrays["MAPPO_FC1_NORM_BIAS"])
    w2 = arrays["MAPPO_FC2_WEIGHT"].reshape(hidden, hidden)
    x = np.maximum(x @ w2.T + arrays["MAPPO_FC2_BIAS"], 0.0)
    x = layer_norm(x, arrays["MAPPO_FC2_NORM_WEIGHT"], arrays["MAPPO_FC2_NORM_BIAS"])
    wa = arrays["MAPPO_ACTION_WEIGHT"].reshape(1, hidden)
    return x @ wa.T + arrays["MAPPO_ACTION_BIAS"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MAPPO header export")
    parser.add_argument("actor", type=Path)
    parser.add_argument("header", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()

    actor_path = args.actor.resolve()
    saved = load_saved_arguments(actor_path)
    env_args = parse_training_args(saved, get_config())
    env_args.algorithm_name = "mappo"
    env_args.use_recurrent_policy = False
    env_args.use_naive_recurrent_policy = False
    env_class = (
        CPLDCMicrogridEnv
        if "--dc_cpl_power_min_w" in saved
        else DCMicrogridEnv
    )
    env = env_class(env_args)
    actor, actor_args = build_actor(actor_path, env)

    rng = np.random.RandomState(args.seed)
    obs_dim = env.observation_space[0].shape[0]
    obs = rng.uniform(-0.5, 0.5, size=(args.samples, obs_dim)).astype(np.float32)
    rnn = np.zeros((args.samples, actor_args.recurrent_N, actor_args.hidden_size), np.float32)
    masks = np.ones((args.samples, 1), np.float32)
    with torch.no_grad():
        torch_action, _, _ = actor(obs, rnn, masks, deterministic=True)
    expected = torch_action.cpu().numpy()
    actual = exported_forward(obs, parse_header(args.header.resolve()))
    maximum_error = float(np.max(np.abs(expected - actual)))
    print(f"Samples   : {args.samples}")
    print(f"Max error : {maximum_error:.9e}")
    print(f"Tolerance : {args.tolerance:.9e}")
    if maximum_error > args.tolerance:
        print("FAILED: exported inference does not match PyTorch")
        return 1
    print("PASSED: exported inference matches PyTorch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


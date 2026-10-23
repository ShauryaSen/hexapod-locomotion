# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env construction for deployments. Import only after ``AppLauncher`` has started Kit."""

from __future__ import annotations

import os

import yaml

from .cfg import DeployCfg

# Every scalar top-level field of <run>/params/env.yaml is copied onto the deployment cfg (reward weights, terrain
# version and its tread / platform / cap, gait clock, action scale, ...), so the env and above all the reward the
# updater optimizes are those pi_0 was trained with; then the deployment overrides below are applied. Measured
# 2026-10-23: the earlier whitelist missed feet_air_time_threshold (trained 0.0, cfg default 0.2), terrain_version
# (trained 2 for the v2.1 seeds, default 1) and the swing-clearance reward of the v2.1 seeds.
SKIP_KEYS = {"seed", "log_dir", "episode_length_s", "num_envs"}


def deploy_env_cfg(cfg: DeployCfg, checkpoint: str, num_envs: int, device: str, run_dir: str | None = None, dr: bool = False, episode_s: float | None = None):
    """Env cfg for a fixed-command deployment: no time-outs, no curricula, nominal DR unless ``dr``."""
    from isaaclab_tasks.utils import parse_env_cfg

    run_dir = run_dir or os.path.dirname(checkpoint)
    env_cfg = parse_env_cfg(cfg.task, device=device, num_envs=num_envs)
    params_path = os.path.join(run_dir, "params", "env.yaml")
    applied = {}
    train_cmd_max = None
    if os.path.exists(params_path):
        params = yaml.load(open(params_path), Loader=yaml.UnsafeLoader)
        for k, v in params.items():
            if k in SKIP_KEYS or not hasattr(env_cfg, k):
                continue
            if isinstance(v, (bool, int, float, str)) or v is None:
                if getattr(env_cfg, k) != v:
                    applied[k] = v
                setattr(env_cfg, k, v)
        train_cmd_max = params.get("command_lin_vel_max")
    else:
        applied["WARNING"] = f"no {params_path}: cfg defaults used (reward may differ from training)"
    for ov in cfg.env_override:
        k, v = ov.split("=", 1)
        setattr(env_cfg, k, yaml.safe_load(v))
        applied[k] = yaml.safe_load(v)
    env_cfg.seed = cfg.env_seed
    env_cfg.spread_resets = False
    env_cfg.terrain_curriculum = False
    env_cfg.command_curriculum = False
    env_cfg.dr.enabled = dr
    env_cfg.command_forward_only = True
    env_cfg.command_backward_fraction = 0.0
    # The command ranges stay as trained: the reward's tracking tolerances are tracking_sigma_ratio x
    # command_lin_vel_max and x command_yaw_rate_max, so pinning the ranges to (0.28, 0) changed the reward (lin
    # tolerance 0.14 vs the trained 0.175; yaw tolerance 0, i.e. the yaw-tracking term ~0 instead of positive). The
    # command itself is pinned to cmd_speed forward / zero yaw after every reset (pin_commands).
    if train_cmd_max is not None and cfg.cmd_speed > float(train_cmd_max) + 1e-9:
        raise ValueError(f"deployment command {cfg.cmd_speed} above the trained range max {train_cmd_max}")
    # no time-outs inside the run: the episode is longer than the deployment (a reposition is an explicit reset)
    env_cfg.episode_length_s = float(episode_s if episode_s is not None else 24 * 3600.0)
    if env_cfg.terrain.terrain_type == "generator":
        env_cfg.terrain.max_init_terrain_level = None
    return env_cfg, applied


def make_env(cfg: DeployCfg, checkpoint: str, num_envs: int, device: str, run_dir: str | None = None, dr: bool = False, episode_s: float | None = None):
    import gymnasium as gym
    import torch

    import hexapod  # noqa: F401 (registers the tasks)

    env_cfg, applied = deploy_env_cfg(cfg, checkpoint, num_envs, device, run_dir, dr, episode_s)
    env = gym.make(cfg.task, cfg=env_cfg).unwrapped
    ids = torch.arange(env.num_envs, device=env.device)
    if env._has_terrain_levels:
        levels, types = terrain_cells(env, ids, cfg.levels)
        env.set_terrain_cells(ids, levels, types)
    env.reset()
    pin_commands(env, cfg.cmd_speed)
    return env, applied


def terrain_cells(env, k, levels):
    """Deployment index k -> (level, column): column = k mod n_cols, level = levels[k mod len(levels)], so the first
    n_cols deployments cover every column (every terrain type) and the levels cycle. Fixed 2026-10-23: the earlier
    rule (column = (k // len(levels)) mod n_cols) put 16 deployments on columns 0-2 only (random rough, slope) and
    12 sweep replicates on columns 0-1 (random rough only)."""
    import torch

    n_cols = env._terrain.terrain_origins.shape[1]
    lv = torch.tensor(list(levels), device=env.device)[k % len(levels)]
    return lv, k % n_cols


def pin_commands(env, cmd_speed: float, env_ids=None):
    """Forward along -y at ``cmd_speed``, no yaw (resets resample commands, so call this after every reset)."""
    if env_ids is None:
        env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -cmd_speed, 0.0
    else:
        env._commands[env_ids, 0], env._commands[env_ids, 1], env._commands[env_ids, 2] = 0.0, -cmd_speed, 0.0


def policy_obs(env):
    return env._get_observations()["policy"]


def forward_speed(env):
    """Speed along the commanded direction (body -y) [m/s], as tools/long_run.py and the env's own metric use."""
    return -env._robot.data.root_lin_vel_b.torch[:, 1]


def reset_in_place(env, env_ids, cmd_speed: float):
    """Reset ``env_ids`` to their terrain cell in the default pose (used for repositioning and fall recovery)."""
    import torch

    env._reset_idx(env_ids.to(torch.int32))
    pin_commands(env, cmd_speed, env_ids)

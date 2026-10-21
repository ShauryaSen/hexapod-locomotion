# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env construction for deployments. Import only after ``AppLauncher`` has started Kit."""

from __future__ import annotations

import os

import yaml

from .cfg import DeployCfg

# env cfg keys read from <run>/params/env.yaml (same list as tools/long_run.py and tools/eval_suite.py)
PARAM_KEYS = ("action_scale", "gait_clock_enabled", "gait_period_s", "command_forward_only", "command_yaw_rate_max", "perceptive", "step_cap")


def deploy_env_cfg(cfg: DeployCfg, checkpoint: str, num_envs: int, device: str, run_dir: str | None = None, dr: bool = False, episode_s: float | None = None):
    """Env cfg for a fixed-command deployment: no time-outs, no curricula, nominal DR unless ``dr``."""
    from isaaclab_tasks.utils import parse_env_cfg

    run_dir = run_dir or os.path.dirname(checkpoint)
    env_cfg = parse_env_cfg(cfg.task, device=device, num_envs=num_envs)
    params_path = os.path.join(run_dir, "params", "env.yaml")
    applied = {}
    if os.path.exists(params_path):
        params = yaml.load(open(params_path), Loader=yaml.UnsafeLoader)
        for k in PARAM_KEYS:
            if k in params and isinstance(params[k], (bool, int, float, str)):
                setattr(env_cfg, k, params[k])
                applied[k] = params[k]
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
    env_cfg.command_lin_vel_min = env_cfg.command_lin_vel_max = cfg.cmd_speed
    env_cfg.command_backward_fraction = 0.0
    env_cfg.command_yaw_rate_max = 0.0
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
        n_cols = env._terrain.terrain_origins.shape[1]
        levels = torch.tensor(list(cfg.levels), device=env.device)[ids % len(cfg.levels)]
        types = (ids // len(cfg.levels)) % n_cols
        env.set_terrain_cells(ids, levels, types)
    env.reset()
    pin_commands(env, cfg.cmd_speed)
    return env, applied


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

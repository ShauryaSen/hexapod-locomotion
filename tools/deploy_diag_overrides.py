# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print Hydra overrides that reproduce a training run's env and PPO settings from its saved params (env.yaml,
agent.yaml), with optional DR-range replacements. Used by tools/deploy_diag_finetune.sh (the "is data the
bottleneck?" diagnostic, 2026-10-23): fine-tune pi_0 with the full training setup at a fixed motor strength.
No Isaac Sim needed.

  python tools/deploy_diag_overrides.py --run_dir <run> --dr motor_strength_range=0.6,0.6
"""

import argparse
import os

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--dr", nargs="*", default=[], help="dr field=lo,hi replacements, e.g. motor_strength_range=0.6,0.6")
parser.add_argument("--agent_set", nargs="*", default=[], help="extra agent.key=value overrides (after the saved ones)")
args = parser.parse_args()

SKIP_ENV = {"seed", "log_dir", "num_envs", "observation_space", "action_space", "state_space", "is_finite_horizon", "wait_for_textures", "ui_window_class_type", "rerender_on_reset", "xr", "num_rerenders_on_reset", "log_dir"}
AGENT_ALGO = ("learning_rate", "schedule", "desired_kl", "entropy_coef", "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam", "max_grad_norm", "value_loss_coef", "use_clipped_value_loss")


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    return str(v)


env = yaml.load(open(os.path.join(args.run_dir, "params", "env.yaml")), Loader=yaml.UnsafeLoader)
agent = yaml.load(open(os.path.join(args.run_dir, "params", "agent.yaml")), Loader=yaml.UnsafeLoader)
out = []
for k, v in env.items():
    if k in SKIP_ENV:
        continue
    if isinstance(v, (bool, int, float, str)) or v is None:
        out.append(f"env.{k}={fmt(v)}")
dr = dict(env.get("dr", {}))
for rep in args.dr:
    k, v = rep.split("=", 1)
    dr[k] = tuple(float(x) for x in v.split(","))
for k, v in dr.items():
    out.append(f"env.dr.{k}={fmt(v)}")
for k in AGENT_ALGO:
    if k in agent.get("algorithm", {}):
        out.append(f"agent.algorithm.{k}={fmt(agent['algorithm'][k])}")
for k in ("num_steps_per_env",):
    if k in agent:
        out.append(f"agent.{k}={fmt(agent[k])}")
out += list(args.agent_set)
print(" ".join(out))

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack several rsl_rl checkpoints (e.g. a fine-tune's model_*.pt) into one snapshot file in the format of
tools/deploy_sweep_updater.py (one policy slice per checkpoint, group "<prefix><iterations>"), so that
tools/deploy_eval_policies.py evaluates them all in one batch next to the frozen pi_0 (one <= 30-min GPU job instead
of one eval-suite run per checkpoint). No Isaac Sim needed.

  python tools/deploy_ckpts_to_snapshot.py --base_iter 5199 --robot_h_per_iter 0.0683 --out <dir>/snapshots \\
      <run>/model_5214.pt <run>/model_5230.pt ...
"""

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy.batched import BatchedGaussianActor, load_rsl_rl_checkpoint  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--base_iter", type=int, required=True, help="model number of iteration 0 minus 1 (resume from 5200: model_5200.pt is iteration 1 -> 5199)")
parser.add_argument("--robot_h_per_iter", type=float, default=512 * 24 * 0.02 / 3600)
parser.add_argument("--prefix", default="ft_it")
parser.add_argument("--out", required=True, help="snapshot directory (adapt_000.pt is written)")
args = parser.parse_args()
infos = [load_rsl_rl_checkpoint(c, "cpu") for c in args.checkpoints]
its = [int(re.findall(r"model_(\d+)\.pt", c)[0]) - args.base_iter for c in args.checkpoints]
a = BatchedGaussianActor(len(infos), infos[0]["obs_dim"], tuple(infos[0]["hidden_dims"]), infos[0]["act_dim"])
with torch.no_grad():
    for i, inf in enumerate(infos):
        a.load_rsl_rl(inf["actor"], idx=torch.tensor([i]))
groups = [{"name": f"{args.prefix}{it}", "k": 1} for it in its]
os.makedirs(args.out, exist_ok=True)
torch.save({"adapt_min": 0, "t_s": 0.0, "actor": {k: v.detach().clone() for k, v in a.state_dict().items()}, "group_of_slice": list(range(len(infos))), "replicate_of_slice": [0] * len(infos), "groups": groups,
            "robot_hours_of_slice": [it * args.robot_h_per_iter for it in its], "checkpoints": args.checkpoints}, os.path.join(args.out, "adapt_000.pt"))
print(f"wrote {os.path.join(args.out, 'adapt_000.pt')}: " + ", ".join(f"{g['name']} ({it * args.robot_h_per_iter:.2f} robot-h)" for g, it in zip(groups, its)))

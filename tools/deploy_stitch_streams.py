# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stitch the per-cycle streams of several short frozen-arm runs of the same robots (same num_envs, same cells per
env, nominal, no updates) into one run dir, cycle after cycle, so that offline tools that need long per-robot streams
(deploy_calibrate_cusum's per-deployment baseline) can use jobs of <= 30 min. Valid when every job starts each robot
from its cell's default pose and nothing persists across jobs (frozen policy, no change, no adaptation); a stitched
stream is then a sequence of segments, as a long run's repositions make it (ASSUMED equivalent, 2026-11-09). No
Isaac Sim needed.

  python tools/deploy_stitch_streams.py <run_a> <run_b> ... --out <dir>
"""

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("--out", required=True)
args = parser.parse_args()

arrs, metas = [], []
for r in args.runs:
    z = np.load(os.path.join(r, "streams.npz"))
    arrs.append({k: z[k] for k in z.files})
    metas.append(json.load(open(os.path.join(r, "meta.json"))))
keys = set(arrs[0])
for r, a, m in zip(args.runs, arrs, metas):
    if set(a) != keys:
        raise ValueError(f"{r}: stream keys differ")
    n = a["t_s"].shape[1]
    if n != arrs[0]["t_s"].shape[1]:
        raise ValueError(f"{r}: {n} robots vs {arrs[0]['t_s'].shape[1]}")
    for k in ("levels_per_env", "types_per_env", "checkpoint", "ensemble", "arm"):
        if k in metas[0] and m.get(k) != metas[0].get(k):
            raise ValueError(f"{r}: meta {k} differs from {args.runs[0]}")
    if m.get("arm", "never") != "never":
        raise ValueError(f"{r}: arm {m.get('arm')}; only frozen (never) runs can be stitched")

out = {}
offset = 0.0
for k in keys:
    out[k] = []
for a in arrs:
    for k in keys:
        out[k].append(a[k] + offset if k == "t_s" else a[k])
    offset += float(a["t_s"][-1, 0])
out = {k: np.concatenate(v, axis=0) for k, v in out.items()}
os.makedirs(args.out, exist_ok=True)
np.savez_compressed(os.path.join(args.out, "streams.npz"), **out)
meta = dict(metas[0])
meta["stitched_from"] = args.runs
meta["minutes"] = float(out["t_s"][-1, 0]) / 60.0
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1, default=str)
print(f"[stitch] {len(args.runs)} runs, {out['t_s'].shape[0]} cycles x {out['t_s'].shape[1]} robots = {meta['minutes']:.2f} min per robot -> {args.out}")

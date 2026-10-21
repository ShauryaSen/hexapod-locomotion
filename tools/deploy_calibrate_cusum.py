# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the CUSUM threshold h to a target false-alarm rate on held-out nominal residual streams. Inputs: the
ensemble's calibration streams (calib_streams.npz, fresh nominal rollouts) and/or never-arm deployment runs without
changes (streams.npz with z). No Isaac Sim needed.

  python tools/deploy_calibrate_cusum.py --ensemble <ensemble.pt> --streams <dir>/calib_streams.npz <run_dir> ... --target 1.0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.detectors import calibrate_cusum_h, three_sigma_fa_rate  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ensemble", required=True, help="ensemble.pt (for nominal mu / sigma)")
parser.add_argument("--streams", nargs="+", required=True, help="calib_streams.npz files and/or deployment run dirs (streams.npz)")
parser.add_argument("--target", type=float, default=None, help="target false alarms per hour (default DetectorCfg)")
parser.add_argument("--k", type=float, default=None)
parser.add_argument("--cycle_s", type=float, default=0.4)
parser.add_argument("--out", default=None, help="default: <ensemble dir>/cusum_calibration.json")
args = parser.parse_args()

det = C.DetectorCfg()
target = args.target if args.target is not None else det.cusum_target_fa_per_hour
k = args.k if args.k is not None else det.cusum_k
ck = torch.load(args.ensemble, map_location="cpu", weights_only=False)
mu = float(ck["state_dict"]["nominal_mu"])
sigma = float(ck["state_dict"]["nominal_sigma"])
zs = []
sources = []
for p in args.streams:
    if os.path.isdir(p):
        z = np.load(os.path.join(p, "streams.npz"))["residual"]
        sources.append(("run", p, z.shape))
    else:
        z = np.load(p)["residual"]
        sources.append(("calib", p, z.shape))
    z = (z - mu) / sigma
    zs.extend([z[:, i] for i in range(z.shape[1])])
cal = calibrate_cusum_h(zs, args.cycle_s, k, target)
cal.update({"nominal_mu": mu, "nominal_sigma": sigma, "sources": [list(s) for s in sources], "n_streams": len(zs),
            "three_sigma_fa_per_hour": {f"x{m}": three_sigma_fa_rate(zs, args.cycle_s, det.three_sigma_k, m) for m in (1, 2, 3, 5)},
            "z_stats": {"mean": float(np.nanmean(np.concatenate(zs))), "std": float(np.nanstd(np.concatenate(zs))), "p99": float(np.nanpercentile(np.concatenate(zs), 99)), "max": float(np.nanmax(np.concatenate(zs)))}})
out = args.out or os.path.join(os.path.dirname(args.ensemble), "cusum_calibration.json")
json.dump(cal, open(out, "w"), indent=1)
print(json.dumps({k2: v for k2, v in cal.items() if k2 != "grid"}, indent=1))
print("grid (h, FA/h):", [(h, round(r, 2)) for h, r in cal["grid"] if r > 0][:12], "...")
print("[cusum] wrote", out)

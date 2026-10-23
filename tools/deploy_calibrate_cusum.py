# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the CUSUM threshold h to a target false-alarm rate on held-out nominal residual streams. Inputs: the
ensemble's calibration streams (calib_streams.npz, fresh nominal rollouts) and/or never-arm deployment runs without
changes (streams.npz with the raw residual). z uses the ensemble's transform (EnsembleCfg.residual_transform) and
nominal statistics; --set_calibration_from recomputes those statistics from a calib_streams.npz / run dir and writes
them into the ensemble.pt (no Kit needed). Also reports the 3-sigma rule's false-alarm rates. No Isaac Sim needed.

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
parser.add_argument("--set_calibration_from", default=None, help="calib_streams.npz or run dir whose residual stream sets the ensemble's nominal statistics (raw and log), written back into ensemble.pt")
parser.add_argument("--transform", default=None, help="override EnsembleCfg.residual_transform (log | none) and write it into ensemble.pt")
parser.add_argument("--baseline_cycles", type=int, default=None, help="R5: reproduce the runner's per-deployment baseline: the first N cycles of each stream set its mean (no alarms there), z = (x - mu_deployment) / sigma_global on the rest. Default DetectorCfg.baseline_cycles; 0 = global mean")
args = parser.parse_args()

det = C.DetectorCfg()
target = args.target if args.target is not None else det.cusum_target_fa_per_hour
k = args.k if args.k is not None else det.cusum_k
from hexapod.deploy.ensemble import DynamicsEnsemble  # noqa: E402

ens = DynamicsEnsemble.load(args.ensemble, "cpu")
if args.transform:
    ens.cfg.residual_transform = args.transform
if args.set_calibration_from:
    p = args.set_calibration_from
    npz = np.load(os.path.join(p, "streams.npz") if os.path.isdir(p) else p)
    ens.set_calibration(npz["residual"], npz["disagreement"] if "disagreement" in npz.files else None)
if args.transform or args.set_calibration_from:
    ck = torch.load(args.ensemble, map_location="cpu", weights_only=False)
    ens.save(args.ensemble, extra=ck.get("extra"))
    print("[cusum] wrote calibration into", args.ensemble)
zs = []
sources = []
for p in args.streams:
    if os.path.isdir(p):
        r = np.load(os.path.join(p, "streams.npz"))["residual"]
        sources.append(("run", p, r.shape))
    else:
        r = np.load(p)["residual"]
        sources.append(("calib", p, r.shape))
    nb = det.baseline_cycles if args.baseline_cycles is None else args.baseline_cycles
    if nb > 0:
        x = np.log(np.clip(r, 1e-12, None)) if ens.cfg.residual_transform == "log" else r
        sd = float(ens.nominal_sigma_log) if ens.cfg.residual_transform == "log" else float(ens.nominal_sigma)
        for i in range(x.shape[1]):
            col = x[:, i]
            valid = np.where(~np.isnan(col))[0]
            if len(valid) <= nb:
                continue
            mu_i = col[valid[:nb]].mean()
            zi = (col - mu_i) / sd
            zi[: valid[nb - 1] + 1] = np.nan  # no alarms while the baseline is measured
            zs.append(zi)
    else:
        z = ens.standardize_np(r)
        zs.extend([z[:, i] for i in range(z.shape[1])])
cal = calibrate_cusum_h(zs, args.cycle_s, k, target, h_grid=np.concatenate([np.arange(0.5, 10.0, 0.25), np.arange(10.0, 200.0, 2.0)]))
cal.update({"baseline_cycles": det.baseline_cycles if args.baseline_cycles is None else args.baseline_cycles, "transform": ens.cfg.residual_transform, "nominal_mu": float(ens.nominal_mu), "nominal_sigma": float(ens.nominal_sigma), "nominal_mu_log": float(ens.nominal_mu_log), "nominal_sigma_log": float(ens.nominal_sigma_log), "sources": [list(s) for s in sources], "n_streams": len(zs),
            "three_sigma_fa_per_hour": {f"x{m}": three_sigma_fa_rate(zs, args.cycle_s, det.three_sigma_k, m) for m in (1, 2, 3, 5)},
            "z_stats": {"mean": float(np.nanmean(np.concatenate(zs))), "std": float(np.nanstd(np.concatenate(zs))), "p99": float(np.nanpercentile(np.concatenate(zs), 99)), "max": float(np.nanmax(np.concatenate(zs)))}})
out = args.out or os.path.join(os.path.dirname(args.ensemble), "cusum_calibration.json")
json.dump(cal, open(out, "w"), indent=1)
print(json.dumps({k2: v for k2, v in cal.items() if k2 != "grid"}, indent=1))
print("grid (h, FA/h):", [(h, round(r, 2)) for h, r in cal["grid"] if r > 0][:12], "...")
print("[cusum] wrote", out)

"""z sanity for RMA teacher checkpoints (stage 2 kill criterion K3; pure torch, no simulator).

For each checkpoint: encode e over a uniform sample of the training DR box (D1-D3) and report
  - z_std: mean per-dimension std of z over e (collapse if ~0), and any NaN;
  - probe R^2 of a linear map z -> each component of e (how much of e survives in z; informative, not gated);
  - wz_ratio: mean column norm of the first-layer weights on z over that on the policy obs (0 at the switch;
    growing = the actor uses z), for actor and critic.
Prints one JSON line per checkpoint.
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.rma import CheckpointPolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+", help="checkpoints or run directories")
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--every", type=int, default=1, help="with a run dir: every k-th checkpoint")
args = ap.parse_args()

files = []
for p in args.paths:
    if os.path.isdir(p):
        fs = sorted(glob.glob(os.path.join(p, "model_*.pt")), key=lambda f: int(re.findall(r"model_(\d+)", f)[0]))
        files += fs[:: args.every] + ([fs[-1]] if fs and fs[-1] not in fs[:: args.every] else [])
    else:
        files.append(p)

g = torch.Generator().manual_seed(0)
lo = torch.tensor([-0.75, 0.5, 0.8])
hi = torch.tensor([0.75, 1.25, 1.2])
e = lo + (hi - lo) * torch.rand(args.n, 3, generator=g)
names = ("dmass", "friction", "strength")


def r2(z, y):
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    sol = torch.linalg.lstsq(X, y.unsqueeze(1)).solution
    res = y - (X @ sol).squeeze(1)
    return 1.0 - res.var().item() / y.var().item()


def wz_ratio(sd, z_dim):
    w = sd["mlp.0.weight"]
    cn = w.norm(dim=0)
    return (cn[-z_dim:].mean() / cn[:-z_dim].mean()).item()


for f in files:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    a = CheckpointPolicy(ck["actor_state_dict"])
    if not a.is_teacher:
        print(json.dumps({"ckpt": os.path.basename(f), "teacher": False}))
        continue
    with torch.no_grad():
        z = a.z(e)
    z_dim = z.shape[1]
    c = CheckpointPolicy(ck["critic_state_dict"])
    with torch.no_grad():
        zc = c.z(e)
    row = {
        "ckpt": os.path.basename(f),
        "iter": ck.get("iter"),
        "nan": bool(torch.isnan(z).any() or torch.isnan(zc).any()),
        "z_std": round(z.std(0).mean().item(), 4),
        "z_abs_max": round(z.abs().max().item(), 3),
        "r2_actor": {n: round(r2(z, e[:, i]), 3) for i, n in enumerate(names)},
        "r2_critic": {n: round(r2(zc, e[:, i]), 3) for i, n in enumerate(names)},
        "wz_ratio_actor": round(wz_ratio(ck["actor_state_dict"], z_dim), 4),
        "wz_ratio_critic": round(wz_ratio(ck["critic_state_dict"], z_dim), 4),
    }
    print(json.dumps(row), flush=True)

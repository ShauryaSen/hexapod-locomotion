"""Make the stage-2 (RMA teacher) init checkpoint from a stage-1 checkpoint (decisions.md P7).

Writes <logs>/rsl_rl/hexapod_rough_direct/<out_run>/model_<iter>.pt with zero-padded z columns in the first layer of
actor and critic, a fresh encoder, and an empty Adam state; train with
    --task Hexapod-Rough-RMATeacher-v0 --resume --load_run <out_run> --checkpoint model_<iter>.pt
Pure torch (no simulator). Also checks that the padded actor reproduces the stage-1 actions exactly.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.rma import CheckpointPolicy, pad_stage1_checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="stage-1 checkpoint (.pt)")
ap.add_argument("--out_dir", required=True, help="new run directory to write the init checkpoint into")
ap.add_argument("--z_dim", type=int, default=8)
ap.add_argument("--seed", type=int, default=0, help="encoder init seed")
args = ap.parse_args()

src = torch.load(args.src, map_location="cpu", weights_only=False)
out = pad_stage1_checkpoint(src, z_dim=args.z_dim, seed=args.seed)
out["infos"]["rma_init_src"] = os.path.abspath(args.src)

# check: same actions as stage 1 for random obs and any e
a0 = CheckpointPolicy(src["actor_state_dict"])
a1 = CheckpointPolicy(out["actor_state_dict"])
x = torch.randn(256, src["actor_state_dict"]["mlp.0.weight"].shape[1])
e = torch.rand(256, 3) * torch.tensor([1.5, 0.75, 0.4]) + torch.tensor([-0.75, 0.5, 0.8])
d = (a0(x) - a1({"policy": x, "privileged": e})).abs().max().item()
print(f"[rma_init] max |action diff| padded vs stage 1: {d:.3e}")
assert d < 1e-6, "padding changed the policy"
z = a1.z(e)
print(f"[rma_init] init z std over e: {z.std(0).mean().item():.3f}; params in optimizer: "
      f"{len(out['optimizer_state_dict']['param_groups'][0]['params'])}")

os.makedirs(args.out_dir, exist_ok=True)
dst = os.path.join(args.out_dir, f"model_{src['iter']}.pt")
torch.save(out, dst)
print(f"[rma_init] wrote {dst} (iter {src['iter']})")

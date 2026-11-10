# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay success test verdict (harness-report "Replay updater", rule fixed 2026-11-03 before any run), from
deploy_eval_policies JSONs of the snapshots (any split into jobs) and a frozen reference:

  PASS if in both blocks (even / odd replicates, 6 deployments each) replay's success at 60 min (mean of the 50- and
  60-min snapshots) minus frozen >= bar (0.35 G_5200 = 0.148), and the paired replay - onpolicy difference (per
  replicate, 12 pairs, same 50/60 mean) has a 95% bootstrap CI above 0. Partial = one block. Fail = neither.

Also prints the per-minute curve per group and block. No Isaac Sim needed.

  python tools/deploy_replay_verdict.py --frozen <eval>/r3eval_frozen.json --evals <eval>/r3eval_m*.json --bar 0.148
"""

import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--frozen", required=True, help="eval JSON whose 'frozen' policy is the reference")
parser.add_argument("--evals", nargs="+", required=True)
parser.add_argument("--bar", type=float, default=0.148)
parser.add_argument("--final_minutes", type=int, nargs="+", default=[50, 60])
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--out", default=None)
args = parser.parse_args()
rng = np.random.default_rng(0)

fr = [p for p in json.load(open(args.frozen))["policies"] if p["group"] == "frozen"][0]
frozen = fr["success"]
succ = {}  # (group, replicate, minute) -> success over the 36 cells
frozen_in_batch = []
for f in args.evals:
    for p in json.load(open(f))["policies"]:
        if p["group"] == "frozen":
            frozen_in_batch += p["success_list"]
            continue
        succ[(p["group"], p["replicate"], p["adapt_min"])] = p["success"]
groups = sorted({k[0] for k in succ})
reps = sorted({k[1] for k in succ})
mins = sorted({k[2] for k in succ})


def final(g, r):
    v = [succ.get((g, r, m)) for m in args.final_minutes]
    return float(np.mean(v)) if all(x is not None for x in v) else None


lines = [f"frozen reference: {frozen:.3f} ({fr['n']} episodes); frozen in the snapshot batches: {np.mean(frozen_in_batch):.3f} ({len(frozen_in_batch)} episodes)", ""]
lines.append("| group | block | " + " | ".join(f"{m} min" for m in mins) + " |")
lines.append("|---|---|" + "---|" * len(mins))
for g in groups:
    for b in (0, 1):
        rb = [r for r in reps if r % 2 == b]
        cells = []
        for m in mins:
            v = [succ[(g, r, m)] for r in rb if (g, r, m) in succ]
            cells.append(f"{np.mean(v):.3f} [{min(v):.2f}-{max(v):.2f}]" if v else "-")
        lines.append(f"| {g} | {'even' if b == 0 else 'odd'} | " + " | ".join(cells) + " |")
res = {"frozen": frozen, "frozen_in_batch": float(np.mean(frozen_in_batch)) if frozen_in_batch else None, "bar": args.bar, "blocks": {}}
blocks_pass = 0
ready = "replay" in groups and all(final("replay", r) is not None for r in reps)
if ready:
    lines.append("")
    for b in (0, 1):
        rb = [r for r in reps if r % 2 == b]
        gain = float(np.mean([final("replay", r) for r in rb])) - frozen
        per = [round(final("replay", r) - frozen, 3) for r in rb]
        ok = gain >= args.bar
        blocks_pass += ok
        res["blocks"]["even" if b == 0 else "odd"] = {"replay_minus_frozen": gain, "per_deployment": per, "pass": ok}
        lines.append(f"block {'even' if b == 0 else 'odd'}: replay - frozen at the {args.final_minutes}-min mean = {gain:+.3f} (bar {args.bar:+.3f}) -> {'pass' if ok else 'miss'}; per deployment {per}")
    if "onpolicy" in groups and all(final("onpolicy", r) is not None for r in reps):
        d = np.array([final("replay", r) - final("onpolicy", r) for r in reps])
        bs = d[rng.integers(0, len(d), size=(args.boot, len(d)))].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        res["paired"] = {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "per_replicate": d.round(3).tolist()}
        lines.append(f"paired replay - onpolicy (12 replicates): {d.mean():+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] -> {'above 0' if lo > 0 else 'not above 0'}")
        ci_ok = lo > 0
        verdict = "PASS" if blocks_pass == 2 and ci_ok else ("PARTIAL" if blocks_pass >= 1 else "FAIL")
        res["verdict"] = verdict
        res["blocks_pass"] = int(blocks_pass)
        res["paired_ci_above_0"] = bool(ci_ok)
        lines.append(f"VERDICT: {verdict} (blocks passing {blocks_pass}/2, paired CI above 0: {ci_ok})")
print("\n".join(lines))
if args.out:
    json.dump(res, open(args.out + ".json", "w"), indent=1)
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")

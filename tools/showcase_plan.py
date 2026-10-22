# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan the showcase recordings from eval-suite JSONs (plain Python, no Isaac).

Reads the final terrain evals of the stage-1 seeds, picks the best seed (highest overall success), and for each
terrain type the hardest level whose success is >= --pass_rate at the showcase command. Reads flat speed-sweep evals
and picks the fastest command bucket that passes P3 (success >= --pass_rate) with 0 falls. Prints a JSON plan.

usage: python3 showcase_plan.py --terrain_evals <json>... --flat_evals <json>...
"""

import argparse
import json
import os
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--terrain_evals", nargs="+", required=True)
p.add_argument("--flat_evals", nargs="+", default=[])
p.add_argument("--cmd", type=float, default=0.28)
p.add_argument("--pass_rate", type=float, default=0.75)
p.add_argument("--flat_pass_rate", type=float, default=0.9)
a = p.parse_args()

best = None
for f in a.terrain_evals:
    d = json.load(open(f))
    cells = [c for c in d["cells"] if abs(c["cmd"] - a.cmd) < 1e-6]
    n = sum(c["n"] for c in cells)
    overall = sum(c["success"] * c["n"] for c in cells) / max(n, 1)
    if best is None or overall > best[1]:
        best = (f, overall, cells, d["checkpoint"])
f, overall, cells, ckpt = best
by = defaultdict(dict)
for c in cells:
    by[c["type"]][c["level"]] = c
hardest = {}
for t, lv in by.items():
    ok = [l for l, c in lv.items() if c["success"] >= a.pass_rate]
    if ok:
        L = max(ok)
        hardest[t] = {"level": L, "success": lv[L]["success"], "falls": lv[L]["falls"], "mean_v": lv[L]["mean_v"]}
flat = None
for ff in a.flat_evals:
    if not os.path.exists(ff):
        continue
    d = json.load(open(ff))
    for c in d["cells"]:
        if c["success"] >= a.flat_pass_rate and c["falls"] == 0.0:
            if flat is None or c["cmd"] > flat["cmd"]:
                flat = {"checkpoint": d["checkpoint"], "cmd": c["cmd"], "success": c["success"], "mean_v": c["mean_v"], "eval": ff}
print(json.dumps({"best_eval": f, "best_overall": overall, "checkpoint": ckpt, "hardest_passing": hardest, "flat_fastest": flat}, indent=1))

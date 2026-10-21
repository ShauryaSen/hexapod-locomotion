# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collate tools/eval_suite.py JSON outputs of a run directory into one table per checkpoint.

Usage: python3 tools/summarize_evals.py <run_dir> [--tag a] [--pass 0.9]
Prints success per command bucket (and per terrain level when present) for every checkpoint, and the top
learnable speed per checkpoint = fastest command bucket whose success rate >= --pass (default 0.9, the L0 bar of
decisions.md P4; the brief says "passing the success definition" without a rate, so 0.9 is a labelled choice).
Plain Python, runs anywhere.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--tag", default="")
parser.add_argument("--pass_rate", type=float, default=0.9)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.run_dir, "eval", f"model_*{('_' + args.tag) if args.tag else ''}.json")),
               key=lambda p: int(re.findall(r"model_(\d+)", os.path.basename(p))[0]))
rows = []
for f in files:
    d = json.load(open(f))
    it = int(re.findall(r"model_(\d+)", os.path.basename(f))[0])
    by_cmd, by_lvl = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for c in d["cells"]:
        by_cmd[c["cmd"]][0] += c["success"] * c["n"]; by_cmd[c["cmd"]][1] += c["n"]
        by_lvl[c["level"]][0] += c["success"] * c["n"]; by_lvl[c["level"]][1] += c["n"]
    cmds = sorted(by_cmd)
    succ = {k: v[0] / v[1] for k, v in by_cmd.items()}
    top = max([k for k in cmds if succ[k] >= args.pass_rate], default=None)
    lvl = {k: v[0] / v[1] for k, v in by_lvl.items()}
    rows.append((it, succ, top, lvl))
cmds = sorted({k for _, s, _, _ in rows for k in s})
levels = sorted({k for _, _, _, l in rows for k in l})
print("| iteration | " + " | ".join(f"{c:.2f}" for c in cmds) + " | top speed (pass >= %.2f) |" % args.pass_rate + ("" if len(levels) < 2 else " " + " | ".join(f"L{l}" for l in levels) + " |"))
print("|---|" + "---|" * (len(cmds) + 1) + ("" if len(levels) < 2 else "---|" * len(levels)))
for it, succ, top, lvl in rows:
    line = f"| {it} | " + " | ".join(f"{succ.get(c, float('nan')):.2f}" for c in cmds) + f" | {top if top is not None else '-'} |"
    if len(levels) >= 2:
        line += " " + " | ".join(f"{lvl.get(l, float('nan')):.2f}" for l in levels) + " |"
    print(line)

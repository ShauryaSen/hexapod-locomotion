# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the showcase clip set from a plan written by tools/showcase_plan.py (host-side driver, plain Python).

Runs tools/record_video.py through tools/spark_run.sh once per clip, sequentially, from ~/src/isaaclab-learning on
the Spark, and writes manifest.json (clip name, terrain, level, column, camera, command, robots, falls and mean speed
reported by the recorder, checkpoint).

usage (on the Spark host, from ~/src/isaaclab-learning):
  python3 hexapod/tools/showcase_record.py --plan logs/videos/showcase/plan.json --out logs/videos/showcase
"""

import argparse
import json
import os
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--out", required=True)
p.add_argument("--cmd", type=float, default=0.28)
a = p.parse_args()
plan = json.load(open(a.plan))
ROOT = os.path.expanduser("~/src/isaaclab-learning")
COL = {"random_rough": 1, "slope": 3, "slope_inv": 5, "stairs": 7, "stairs_inv": 9, "boxes": 11}  # v2.1 higher column per type
host_ckpt = plan["checkpoint"]  # container path /workspace/isaaclab/...
manifest = []


def rec(name, task, ckpt, cmd, extra, meta):
    log = f"logs/videos/showcase_logs/{name}.log"
    os.makedirs(os.path.join(ROOT, "logs/videos/showcase_logs"), exist_ok=True)
    out = f"/workspace/isaaclab/{a.out.replace('logs/', '', 1)}/{name}"
    cmdline = ["hexapod/tools/spark_run.sh", log, "--", "env", "PYTHONPATH=/workspace/learning", "./isaaclab.sh", "-p",
               "/workspace/learning/hexapod/tools/record_video.py", "--headless", "--enable_cameras", "--task", task,
               "--checkpoint", ckpt, "--cmd", str(cmd), "--out", out] + extra
    env = dict(os.environ, SPARK_RUN_ATTEMPTS="6", SPARK_RUN_STALL_S="0")
    subprocess.run(cmdline, cwd=ROOT, env=env)
    txt = open(os.path.join(ROOT, log)).read() if os.path.exists(os.path.join(ROOT, log)) else ""
    m = re.search(r"\[record_video\] (\d+) steps, (\d+) envs, cmd ([\d.]+), falls (\d+), mean speed along cmd ([-\d.]+)", txt)
    entry = dict(meta, name=name, task=task, checkpoint=ckpt, cmd=cmd, args=extra)
    if m:
        entry.update(steps=int(m.group(1)), robots=int(m.group(2)), falls=int(m.group(4)), mean_speed=float(m.group(5)))
    else:
        entry.update(error="no result line")
    manifest.append(entry)
    json.dump(manifest, open(os.path.join(ROOT, a.out, "manifest.json"), "w"), indent=1)
    print(f"[showcase] {name}: {entry.get('falls', '?')} falls, {entry.get('mean_speed', '?')} m/s", flush=True)


os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
R = "Hexapod-Rough-Direct-v0"
# 1. per type at the hardest passing level
for t, info in plan["hardest_passing"].items():
    L, c = info["level"], COL[t]
    base = ["--levels", str(L), "--column", str(c), "--num_envs", "1"]
    rec(f"type_{t}_L{L}_behind", R, host_ckpt, a.cmd, base + ["--steps", "600", "--camera", "behind"], {"type": t, "level": L, "view": "behind", "eval": info})
    if t in ("stairs", "stairs_inv", "boxes"):
        rec(f"type_{t}_L{L}_high", R, host_ckpt, a.cmd, base + ["--steps", "600", "--camera", "high"], {"type": t, "level": L, "view": "high", "eval": info})
# 2. multi-robot L5 per type
for t, c in COL.items():
    rec(f"multi_{t}_L5_high", R, host_ckpt, a.cmd, ["--levels", "5", "--column", str(c), "--num_envs", "7", "--lateral_spread", "0.6", "--steps", "600", "--camera", "high"],
        {"type": t, "level": 5, "view": "high", "robots_planned": 7})
# 4. moving overview
rec("overview_v21_48robots_wide", R, host_ckpt, a.cmd, ["--spread", "--levels", "0", "3", "5", "9", "--num_envs", "48", "--steps", "750", "--cam_eye", "20", "-18", "14", "--cam_lookat", "0", "0", "-1"],
    {"type": "all", "level": "0/3/5/9", "view": "wide"})
# 3. flat fastest
fl = plan.get("flat_fastest")
if fl:
    F = "Hexapod-Flat-Direct-v0"
    v = fl["cmd"]
    side = ["--num_envs", "1", "--steps", "600", "--cam_eye", "2.0", "-2.5", "0.7", "--cam_lookat", "0.0", "-2.5", "0.05"]
    behind = ["--num_envs", "1", "--steps", "600", "--cam_eye", "0.0", "1.5", "1.0", "--cam_lookat", "0.0", "-3.0", "0.0"]
    rec(f"flat_fastest_cmd{v:.2f}_side", F, fl["checkpoint"], v, side, {"type": "flat", "view": "side", "eval": fl})
    rec(f"flat_fastest_cmd{v:.2f}_behind", F, fl["checkpoint"], v, behind, {"type": "flat", "view": "behind", "eval": fl})
    rec("flat_same_policy_cmd0.28_side", F, fl["checkpoint"], 0.28, side, {"type": "flat", "view": "side"})
print("[showcase] done", flush=True)

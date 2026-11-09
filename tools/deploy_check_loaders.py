# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict loaders (supervisor 11-08), CPU, no Isaac Sim: SHA256SUMS verified at load, the RMA student only from a
frozen dir, params/env.yaml required, env obs size = policy input. Works on temp copies of the frozen pi_0 dir.

  python tools/deploy_check_loaders.py --frozen /workspace/isaaclab/logs/frozen/pi0_rma_2026-11-08
"""

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.batched import check_obs_dim, load_rsl_rl_checkpoint  # noqa: E402
from hexapod.deploy.envutil import deploy_env_cfg  # noqa: E402
from hexapod.deploy.frozen import verify_listed  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--frozen", required=True)
args = parser.parse_args()
ok = True


def expect_error(name, fn, kind=Exception):
    global ok
    try:
        fn()
    except kind as e:
        print(f"{name}: raised as expected ({type(e).__name__}: {str(e)[:110]})")
        return
    print(f"{name}: DID NOT RAISE")
    ok = False


tmp = tempfile.mkdtemp()
good = os.path.join(tmp, "frozen")
os.makedirs(os.path.join(good, "params"))
for f in ("SHA256SUMS", "student_model_1000.pt", "params/env.yaml"):
    shutil.copyfile(os.path.join(args.frozen, f), os.path.join(good, f))  # contents only: the frozen files are read-only
ck = os.path.join(good, "student_model_1000.pt")

# 1. the frozen copy loads, the checkpoint and env.yaml verify
info = load_rsl_rl_checkpoint(ck, "cpu")
v = verify_listed(os.path.join(good, "params", "env.yaml"))
print(f"[1] frozen copy: student loads (input {info['obs_dim']}), env.yaml verified {v}")
ok &= v and info["rma_student"] is not None

# 2. a changed byte in a listed file -> error
bad = os.path.join(tmp, "bad")
shutil.copytree(good, bad, copy_function=shutil.copyfile)
with open(os.path.join(bad, "params", "env.yaml"), "a") as f:
    f.write("\n# edited\n")
expect_error("[2] edited env.yaml in a frozen dir", lambda: verify_listed(os.path.join(bad, "params", "env.yaml")), RuntimeError)
with open(os.path.join(bad, "student_model_1000.pt"), "r+b") as f:
    f.seek(-1, 2)
    b = f.read(1)
    f.seek(-1, 2)
    f.write(bytes([b[0] ^ 1]))
expect_error("[2] flipped byte in the checkpoint", lambda: load_rsl_rl_checkpoint(os.path.join(bad, "student_model_1000.pt"), "cpu"), RuntimeError)

# 3. the student outside a frozen dir -> error
loose = os.path.join(tmp, "loose")
os.makedirs(loose)
shutil.copyfile(ck, os.path.join(loose, "student_model_1000.pt"))
expect_error("[3] student without SHA256SUMS", lambda: load_rsl_rl_checkpoint(os.path.join(loose, "student_model_1000.pt"), "cpu"), RuntimeError)

# 4. no params/env.yaml -> error (raised before Isaac Lab is imported)
expect_error("[4] run dir without params/env.yaml", lambda: deploy_env_cfg(C.DeployCfg(), os.path.join(loose, "student_model_1000.pt"), 4, "cpu"), FileNotFoundError)

# 5. obs size: 68 passes; 66 (the 11-08 fallback) and a plain policy on the wrong size raise
check_obs_dim(info, 68)
print("[5] obs 68 for the student (68 + z 8 = 76): accepted")
expect_error("[5] obs 66 for the student", lambda: check_obs_dim(info, 66), ValueError)
expect_error("[5] plain policy (input 68) on obs 66", lambda: check_obs_dim({"obs_dim": 68, "rma_student": None}, 66), ValueError)
check_obs_dim({"obs_dim": 68, "rma_student": None}, 68)
shutil.rmtree(tmp)
print("loader checks passed" if ok else "LOADER CHECKS FAILED")
sys.exit(0 if ok else 1)

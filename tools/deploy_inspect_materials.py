# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Which physics materials and friction combine modes are in force in the deployment env? Builds the env (no
stepping beyond the initial reset), then lists every prim with a physics material (UsdPhysics.MaterialAPI):
static/dynamic friction and PhysX friction/restitution combine modes; every collider under /World/ground and
under env 0's robot with its bound physics material (material:binding:physics, direct or inherited), and the
PhysX scene's default material if any. Also prints the robot material buffer that set_friction writes.

Motivation (MEASURED 2026-10-22, deploy_check_friction on the flat L0 slope column): with set_friction(0.05) the
feet's |F_xy|/F_z reaches p99 0.35 / max 0.56 (Coulomb cap would be 0.05 under 'multiply' with ground mu 1.0); the
numbers fit an 'average' combine, (0.05 + 1.0) / 2 = 0.525.

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_inspect_materials.py --headless --checkpoint <model.pt>
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import warp as wp  # noqa: E402
from pxr import Usd, UsdPhysics, UsdShade  # noqa: E402

try:
    from pxr import PhysxSchema  # noqa: E402
except Exception:  # noqa: BLE001
    PhysxSchema = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.envutil import make_env  # noqa: E402

env, applied = make_env(C.DeployCfg(task=args.task), args.checkpoint, args.num_envs, args.device)
stage = env.sim.stage if hasattr(env.sim, "stage") else __import__("omni.usd").usd.get_context().get_stage()


def attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else (f"<default {a.Get()}>" if a else None)


materials = {}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        materials[str(prim.GetPath())] = {
            "static": attr(prim, "physics:staticFriction"), "dynamic": attr(prim, "physics:dynamicFriction"), "restitution": attr(prim, "physics:restitution"),
            "frictionCombineMode": attr(prim, "physxMaterial:frictionCombineMode"), "restitutionCombineMode": attr(prim, "physxMaterial:restitutionCombineMode"),
            "applied_schemas": [str(s) for s in prim.GetAppliedSchemas()],
        }


def bound_material(prim):
    p = prim
    while p and p.IsValid() and str(p.GetPath()) != "/":
        rel = p.GetRelationship("material:binding:physics")
        if rel and rel.GetTargets():
            return str(rel.GetTargets()[0]), str(p.GetPath())
        api = UsdShade.MaterialBindingAPI(p)
        mat, _ = api.ComputeBoundMaterial("physics")
        if mat:
            return str(mat.GetPath()), str(p.GetPath()) + " (computed)"
        p = p.GetParent()
    return None, None


colliders = []
for root in ("/World/ground", "/World/envs/env_0/Robot"):
    rp = stage.GetPrimAtPath(root)
    if not rp or not rp.IsValid():
        colliders.append({"root": root, "missing": True})
        continue
    for prim in Usd.PrimRange(rp):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mat, where = bound_material(prim)
            colliders.append({"collider": str(prim.GetPath()), "type": prim.GetTypeName(), "material": mat, "bound_at": where})
scene_default = []
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene":
        rel = prim.GetRelationship("material:binding:physics")
        scene_default.append({"scene": str(prim.GetPath()), "default_material": [str(t) for t in rel.GetTargets()] if rel else None})
buf = wp.to_torch(env._robot.root_view.get_material_properties())
res = {"materials": materials, "colliders": colliders, "physics_scenes": scene_default,
       "robot_material_buffer_env0": buf[0].tolist(), "robot_material_buffer_shape": list(buf.shape),
       "cfg": {"sim_material": {"static": env.cfg.sim.physics_material.static_friction, "combine": env.cfg.sim.physics_material.friction_combine_mode},
               "terrain_material": {"static": env.cfg.terrain.physics_material.static_friction, "combine": env.cfg.terrain.physics_material.friction_combine_mode}}}
print(json.dumps({"materials": materials, "physics_scenes": scene_default, "cfg": res["cfg"]}, indent=1, default=str))
from collections import Counter  # noqa: E402

print("[materials] colliders by (root, bound material):", Counter((c.get("collider", c.get("root", "")).split("/")[2] + "/" + (c.get("collider", "").split("/")[3] if c.get("collider", "").count("/") > 3 else ""), c.get("material")) for c in colliders))
print("[materials] robot material buffer env 0 (static, dynamic, restitution) per shape:", buf[0].tolist())
out = args.out or os.path.join(os.path.dirname(args.checkpoint), "ensemble", "materials_inspect.json")
json.dump(res, open(out, "w"), indent=1, default=str)
print("[materials] wrote", out)
env.close()
app.close()

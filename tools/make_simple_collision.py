# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace the hexapod's 153 SDF mesh colliders with one primitive per link (19 shapes), fitted to the CAD.

Body: a box = the bounding box of its collision parts. Each leg segment (coxa, femur, tibia): a capsule along the
link's +x axis (the joint-to-joint direction in this URDF), spanning the parts' x extent, with radius from the
cross-section of the distal half of the segment (so the tibia's radius is its shaft/toe, not its joint housing).
The tibia capsule's far end-cap therefore sits at the real toe tip. Visual meshes, masses, inertias, joints and
limits are untouched; the output is a new file next to the original (relative references stay valid).

Run inside the container:
  ./isaaclab.sh -p /workspace/learning/hexapod/tools/make_simple_collision.py [--visible] [--radius_scale 1.0]
"""

import argparse
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="/workspace/learning/hexapod/assets/usd/robot.usda")
parser.add_argument("--out", default=None, help="default: robot_simple.usda (or robot_simple_visible.usda) next to src")
parser.add_argument("--radius_scale", type=float, default=1.0, help="multiplier on the fitted capsule radii")
parser.add_argument("--radius_quantile", type=float, default=0.95, help="cross-section quantile used as the radius")
parser.add_argument("--visible", action="store_true", help="render the shapes (for a visual check); otherwise purpose=guide")
args = parser.parse_args()
out = args.out or os.path.join(os.path.dirname(args.src), "robot_simple_visible.usda" if args.visible else "robot_simple.usda")

stage = Usd.Stage.Open(args.src)
robot = stage.GetPrimAtPath("/Robot")
links = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
xc = UsdGeom.XformCache()
report = []

for link in links:
    col = link.GetChild("collisions")
    parts = list(col.GetChildren())
    pts = []
    for part in parts:
        p = np.array(UsdGeom.Mesh(part).GetPointsAttr().Get(), dtype=np.float64)
        M = np.array(xc.ComputeRelativeTransform(part, link)[0])  # row-vector convention: p_link = [p, 1] @ M
        pts.append((np.c_[p, np.ones(len(p))] @ M)[:, :3])
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    for part in parts:
        stage.RemovePrim(part.GetPath())
    shape_path = col.GetPath().AppendChild("shape")
    if link.GetName() == "body":
        box = UsdGeom.Cube.Define(stage, shape_path)
        box.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(box)
        xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        desc = f"box {np.round(hi - lo, 4)} at {np.round((lo + hi) / 2, 4)}"
        prim = box.GetPrim()
    else:
        distal = pts[pts[:, 0] > (lo[0] + hi[0]) / 2]
        r = float(np.quantile(np.linalg.norm(distal[:, 1:], axis=1), args.radius_quantile)) * args.radius_scale
        x0, x1 = lo[0] + r, hi[0] - r  # end-cap centres; far cap surface = hi[0] (the toe tip for the tibia)
        if x1 <= x0:
            x0 = x1 = (lo[0] + hi[0]) / 2
        cap = UsdGeom.Capsule.Define(stage, shape_path)
        cap.GetAxisAttr().Set("X")
        cap.GetRadiusAttr().Set(r)
        cap.GetHeightAttr().Set(max(x1 - x0, 0.0))
        UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, float((lo[1] + hi[1]) / 2), float((lo[2] + hi[2]) / 2)))
        desc = f"capsule r={r:.4f} x=[{lo[0]:.4f},{hi[0]:.4f}] (cylinder {x1 - x0:.4f})"
        prim = cap.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    gp = UsdGeom.Gprim(prim)
    if args.visible:
        gp.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.3, 0.1)]))
        gp.GetDisplayOpacityAttr().Set(Vt.FloatArray([0.5]))
    else:
        gp.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    report.append(f"{link.GetName():10s} {len(parts):3d} parts -> {desc}")

# The fitted femur capsules (their motor housings, r ~3 cm) overlap the body box at rest. PhysX already ignores
# contacts between links joined by a joint (body-coxa, coxa-femur), but body-femur is two joints apart and would
# generate large spurious forces (measured: motors pinned at the cap, body "contact" at rest). Filter those pairs.
body = stage.GetPrimAtPath("/Robot/body")
femurs = [l.GetPath() for l in links if l.GetName().endswith("_femur")]
UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(femurs)
report.append(f"filtered pairs: body <-> {len(femurs)} femur links")

layer = stage.GetRootLayer()
cd = dict(layer.customLayerData)
cd["collision_recipe"] = f"one primitive per link, fitted by make_simple_collision.py (radius_scale {args.radius_scale}, quantile {args.radius_quantile})"
layer.customLayerData = cd
layer.Export(out)
print("\n".join(report))
check = Usd.Stage.Open(out)
n = sum(1 for p in Usd.PrimRange(check.GetPrimAtPath("/Robot")) if p.HasAPI(UsdPhysics.CollisionAPI))
print(f"wrote {out}: {n} collision shapes")

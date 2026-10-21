# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose spawning on the rough terrain: zero actions, print per-env body contact force, root z, terrain origin z,
toe z and the done flags for the first steps, grouped by terrain type."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--rows", type=int, default=10)
parser.add_argument("--cols", type=int, default=10)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--flat", action="store_true", help="use HexapodFlatEnvCfg (plane) instead of the rough terrain")
parser.add_argument("--usd", default=None, help="robot usd override")
parser.add_argument("--anymal", action="store_true", help="use Isaac Lab ROUGH_TERRAINS_CFG (8 m tiles) instead")
parser.add_argument("--hscale", type=float, default=None, help="override horizontal_scale")
parser.add_argument("--no_grid", action="store_true", help="drop the random-grid mesh sub-terrain")
parser.add_argument("--plane", action="store_true", help="rough cfg but terrain_type=plane")
parser.add_argument("--no_replicate", action="store_true", help="scene.replicate_physics=False")
parser.add_argument("--spacing", type=float, default=None, help="scene.env_spacing override")
parser.add_argument("--z_offset", type=float, default=0.0, help="spawn height added at reset (m)")
parser.add_argument("--no_self_collision", action="store_true")
parser.add_argument("--no_filter", action="store_true", help="cfg.filter_env_collisions=False")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg, HexapodRoughEnvCfg  # noqa: E402

cfg = HexapodFlatEnvCfg() if args.flat else HexapodRoughEnvCfg()
cfg.scene.num_envs = args.num_envs
if not args.flat:
    if args.anymal:
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
        cfg.terrain.terrain_generator = ROUGH_TERRAINS_CFG
    if args.no_grid:
        from hexapod.hexapod_terrains_cfg import make_hexapod_terrains_cfg
        cfg.terrain.terrain_generator = make_hexapod_terrains_cfg(include_random_grid=False)
    if args.hscale is not None:
        cfg.terrain.terrain_generator.horizontal_scale = args.hscale
    cfg.terrain.terrain_generator.num_rows = args.rows
    cfg.terrain.terrain_generator.num_cols = args.cols
    print("terrain generator:", cfg.terrain.terrain_generator)
    if args.plane:
        cfg.terrain.terrain_type = "plane"
        cfg.terrain.terrain_generator = None
if args.no_replicate:
    cfg.scene.replicate_physics = False
if args.spacing is not None:
    cfg.scene.env_spacing = args.spacing
print("scene:", cfg.scene)
if args.usd:
    cfg.robot.spawn.usd_path = args.usd
if args.no_filter:
    cfg.filter_env_collisions = False
if args.no_self_collision:
    cfg.robot.spawn.articulation_props.enabled_self_collisions = False
cfg.gait_clock_enabled = True
cfg.dr.enabled = False
cfg.spread_resets = False
cfg.terrain_curriculum = False
env = HexapodEnv(cfg)
N = env.num_envs
print("terrain types per column:", env.terrain_type_names)
if env._has_terrain_levels:
    print("terrain_levels:", env._terrain.terrain_levels.tolist())
    print("terrain_types:", env._terrain.terrain_types.tolist())
print("usd:", cfg.robot.spawn.usd_path, "sim playing:", env.sim.is_playing(), "dt", env.physics_dt)
print("env_origins z:", [round(v, 4) for v in env._terrain.env_origins[:, 2].tolist()])
if env._has_terrain_levels:
    import numpy as np
    from pxr import Usd, UsdGeom
    import isaaclab.sim as sim_utils
    for name, m in env._terrain.meshes.items():
        print(f"terrain mesh '{name}': {len(m.faces)} faces, bounds {np.round(m.bounds, 3).tolist()}")
        V = np.asarray(m.vertices)
        for e in range(min(N, 4)):
            o = env._terrain.env_origins[e].cpu().numpy()
            near = V[(np.abs(V[:, 0] - o[0]) < 0.06) & (np.abs(V[:, 1] - o[1]) < 0.06)]
            print(f"   env {e}: origin {np.round(o, 3).tolist()}, surface z near origin: {np.round(near[:, 2].min(), 4) if len(near) else None}..{np.round(near[:, 2].max(), 4) if len(near) else None} ({len(near)} verts)")
    stage = sim_utils.get_current_stage() if hasattr(sim_utils, "get_current_stage") else Usd.Stage.Open
    for path in env._terrain.terrain_prim_paths:
        prim = env.sim.stage.GetPrimAtPath(path) if hasattr(env.sim, "stage") else None
        if prim is not None and prim.IsValid():
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            print(f"terrain prim {path}: translate {[round(v, 3) for v in xf.ExtractTranslation()]} type {prim.GetTypeName()} children {[c.GetName() for c in prim.GetChildren()]}")
            for c in prim.GetChildren():
                cxf = UsdGeom.Xformable(c).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                print(f"   child {c.GetName()} ({c.GetTypeName()}): translate {[round(v, 3) for v in cxf.ExtractTranslation()]} scale {[round(v, 3) for v in (cxf.ExtractRotationMatrix() * 1).GetRow(0)]}")
# log what _get_dones computes, but never terminate, so the physical state can be watched
_orig_dones = env._get_dones


def _dones_logged():
    died, timeout = _orig_dones()
    f = env._contact_sensor.data.net_forces_w_history.torch[:, :, env._base_id]
    print(f"   [_get_dones] died {died.int().tolist()} hist shape {tuple(f.shape)} max {f.abs().max().item():.3f} nan {torch.isnan(f).any().item()}")
    return torch.zeros_like(died), timeout


env._get_dones = _dones_logged
if args.z_offset:
    env._robot.data.default_root_pose.torch[:, 2] += args.z_offset
if env._has_terrain_levels:
    import numpy as np
    from pxr import UsdGeom
    mesh_prim = env.sim.stage.GetPrimAtPath("/World/ground/terrain/mesh")
    P = np.asarray(mesh_prim.GetAttribute("points").Get())
    print(f"terrain mesh prim: {len(P)} points, bounds min {np.round(P.min(0), 3).tolist()} max {np.round(P.max(0), 3).tolist()}, faces {len(mesh_prim.GetAttribute('faceVertexCounts').Get())}")
    print("   collision api:", mesh_prim.HasAPI(UsdGeom.Tokens.Xformable) if False else [a for a in mesh_prim.GetAppliedSchemas()], "approximation:", mesh_prim.GetAttribute("physics:approximation").Get())
    for e in range(min(N, 4)):
        o = env._terrain.env_origins[e].cpu().numpy()
        near = P[(np.abs(P[:, 0] - o[0]) < 0.3) & (np.abs(P[:, 1] - o[1]) < 0.3)]
        print(f"   env {e}: origin {np.round(o, 3).tolist()}, mesh z within 0.3 m: {np.round(near[:, 2].min(), 4) if len(near) else None}..{np.round(near[:, 2].max(), 4) if len(near) else None} ({len(near)} pts)")
env.reset()
print("root z after reset:", [round(v, 4) for v in env._robot.data.root_pos_w.torch[:, 2].tolist()])
tib = [env._robot.body_names.index(f"{l}_tibia") for l in ("lf", "lm", "lr", "rf", "rm", "rr")]
for i in range(args.steps):
    obs, rew, died, timeout, _ = env.step(torch.zeros(N, 18, device=env.device))
    f = torch.linalg.norm(env._contact_sensor.data.net_forces_w.torch[:, env._base_id], dim=-1).squeeze(1)
    feet = torch.linalg.norm(env._contact_sensor.data.net_forces_w.torch[:, env._feet_ids], dim=-1)
    und = torch.linalg.norm(env._contact_sensor.data.net_forces_w.torch[:, env._undesired_contact_body_ids], dim=-1).max(dim=1).values
    z = env._robot.data.root_pos_w.torch[:, 2] - env._terrain.env_origins[:, 2]
    toe_z = (env._robot.data.body_link_pos_w.torch[:, tib, 2] - env._terrain.env_origins[:, 2:3]).min(dim=1).values
    if i < 5 or i % 10 == 0:
        hist = torch.linalg.norm(env._contact_sensor.data.net_forces_w_history.torch[:, :, env._base_id], dim=-1).max(dim=1).values.squeeze(1)
        print(f"step {i}: died {died.int().tolist()} body hist max |F| {[round(v, 2) for v in hist.tolist()]} |qdot| {[round(v, 3) for v in env._robot.data.joint_vel.torch.norm(dim=1).tolist()]}")
        print(f"   body |F| {[round(v, 2) for v in f.tolist()]}")
        allf = torch.linalg.norm(env._contact_sensor.data.net_forces_w.torch[0], dim=-1)
        print(f"   env0 |F| per body: {dict(zip(env._contact_sensor.body_names, [round(v, 1) for v in allf.tolist()]))}")
        print(f"   feet in contact {(feet > 1.0).sum(dim=1).tolist()}")
        print(f"   root z - origin z {[round(v, 4) for v in z.tolist()]}  root xy - origin xy {[round(v, 3) for v in (env._robot.data.root_pos_w.torch[:, :2] - env._terrain.env_origins[:, :2]).norm(dim=1).tolist()]}")
        print(f"   min tibia-link z - origin z {[round(v, 4) for v in toe_z.tolist()]}")
env.close()
app.close()

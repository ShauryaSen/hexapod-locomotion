# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit the generated terrain against its spec without a simulation scene.

Builds HEXAPOD_TERRAINS_CFG with Isaac Lab's TerrainGenerator (CPU, no env), rasterizes the mesh vertices to a
0.02 m height map, and writes: a top-down height map PNG with a colour bar [m], cross-sections per type at levels
5 and 9, and a table per type x level {0, 5, 9}: max |z - median|, p95 |dz| between neighbouring cells, max slope
angle over 0.1 m, top-5% height (obstacle / block height). Spec: noise 0.5-2 cm scaled by level, slopes <= 15 deg,
obstacles and grid blocks <= step cap, platform 0.5 m, border 5 m.

Run inside the container (Kit starts for the imports, no scene):
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/terrain_audit.py --headless \
      --out /workspace/isaaclab/logs/terrain_audit [--step_cap 0.0085]
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--step_cap", type=float, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--version", type=int, default=1)
parser.add_argument("--dump_tile", nargs=2, default=None, metavar=("TYPE", "LEVEL"), help="print a coarse 2-D height grid of one tile (mm, relative to the origin)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hexapod.hexapod_terrains_cfg import HEXAPOD_TERRAINS_CFG, STEP_CAP, make_hexapod_terrains_cfg, terrain_type_names  # noqa: E402

cfg = make_hexapod_terrains_cfg(step_cap=args.step_cap, version=args.version) if (args.step_cap is not None or args.version != 1) else HEXAPOD_TERRAINS_CFG
cfg.seed = args.seed
gen = cfg.class_type(cfg=cfg, device="cpu")
mesh = gen.terrain_mesh
origins = np.asarray(gen.terrain_origins)  # (rows, cols, 3)
names = terrain_type_names(cfg)
V = np.asarray(mesh.vertices)
res = cfg.horizontal_scale
x0, y0 = V[:, 0].min(), V[:, 1].min()
nx, ny = int((V[:, 0].max() - x0) / res) + 2, int((V[:, 1].max() - y0) / res) + 2
H = np.full((nx, ny), -np.inf)
ix, iy = ((V[:, 0] - x0) / res).round().astype(int), ((V[:, 1] - y0) / res).round().astype(int)
np.maximum.at(H, (ix, iy), V[:, 2])  # highest vertex per cell (box tops win)
H[np.isinf(H)] = np.nan
# fill cells without vertices (box interiors, border) from the nearest filled cell
from scipy import ndimage  # noqa: E402

mask = np.isnan(H)
idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
Hf = H[tuple(idx)]
print(f"[audit] mesh {len(mesh.faces)} faces, bounds {np.round(mesh.bounds, 3).tolist()}, grid {nx}x{ny} at {res} m, {mask.mean():.1%} cells filled by nearest")

os.makedirs(args.out, exist_ok=True)
# ---- top-down height map
fig, ax = plt.subplots(figsize=(10, 9))
ext = [y0, y0 + ny * res, x0, x0 + nx * res]
im = ax.imshow(Hf, origin="lower", extent=ext, cmap="terrain", vmin=np.nanpercentile(Hf, 0.5), vmax=np.nanpercentile(Hf, 99.5))
plt.colorbar(im, ax=ax, label="height [m]")
for r in range(cfg.num_rows):
    for c in range(cfg.num_cols):
        o = origins[r, c]
        ax.text(o[1], o[0], f"L{r}\n{names[c][:6]}", ha="center", va="center", fontsize=5, color="k")
ax.set_xlabel("y [m] (robots walk towards -y)")
ax.set_ylabel("x [m] (rows = curriculum levels)")
ax.set_title(f"HEXAPOD_TERRAINS_CFG: {cfg.num_rows} levels x {cfg.num_cols} columns, {cfg.size[0]} m tiles, border {cfg.border_width} m, step cap {args.step_cap or STEP_CAP} m")
fig.savefig(os.path.join(args.out, "terrain_heightmap.png"), dpi=150, bbox_inches="tight")

# ---- exact per-tile heights by vertical ray casts on the cropped mesh (the vertex binning above is only for the
# overview picture: box meshes have vertices at corners only, so nearest-fill distorts stairs and boxes)
F_centroid = V[mesh.faces].mean(axis=1)


def tile_heights(o, n=100):
    xs, ys = o[0] - size / 2, o[1] + -size / 2
    sel = (F_centroid[:, 0] >= xs - 0.1) & (F_centroid[:, 0] <= xs + size + 0.1) & (F_centroid[:, 1] >= ys - 0.1) & (F_centroid[:, 1] <= ys + size + 0.1)
    sub = mesh.submesh([np.where(sel)[0]], append=True)
    cx = xs + (np.arange(n) + 0.5) * size / n
    cy = ys + (np.arange(n) + 0.5) * size / n
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    orig = np.stack([gx.ravel(), gy.ravel(), np.full(n * n, 5.0)], axis=1)
    dirs = np.tile([0.0, 0.0, -1.0], (n * n, 1))
    loc, idx_ray, _ = sub.ray.intersects_location(orig, dirs, multiple_hits=True)
    H2 = np.full(n * n, np.nan)
    np.maximum.at(H2, idx_ray, loc[:, 2]) if len(idx_ray) else None
    H2[np.isinf(H2)] = np.nan
    return H2.reshape(n, n), size / n


# ---- per tile stats and cross-sections
size = cfg.size[0]
first_col = {}
for c, n in enumerate(names):
    first_col.setdefault(n, c)
rows_tbl = []
fig, axes = plt.subplots(len(first_col), 2, figsize=(12, 2.2 * len(first_col)), sharex=True)
for ti, (name, c) in enumerate(first_col.items()):
    for li, lvl in enumerate((0, 5, 9)):  # noqa: B007
        o = origins[lvl, c]
        xs, xe = o[0] - size / 2, o[0] + size / 2
        ys, ye = o[1] - size / 2, o[1] + size / 2
        sl, res_t = tile_heights(o)
        sl = np.where(np.isnan(sl), np.nanmedian(sl), sl)
        z = sl - np.median(sl)
        dz = np.maximum(np.abs(np.diff(sl, axis=0)).max(initial=0), 0)
        dz_all = np.concatenate([np.abs(np.diff(sl, axis=0)).ravel(), np.abs(np.diff(sl, axis=1)).ravel()])
        k = int(0.1 / res_t)
        slope = max(np.abs(sl[k:] - sl[:-k]).max() / 0.1, np.abs(sl[:, k:] - sl[:, :-k]).max() / 0.1)
        rows_tbl.append({
            "type": name, "level": lvl, "max_abs_z_m": float(np.abs(z).max()), "p95_dz_m": float(np.percentile(dz_all, 95)),
            "max_dz_m": float(dz_all.max()), "max_slope_deg": float(math.degrees(math.atan(slope))),
            "p95_height_m": float(np.percentile(z, 95)), "p5_height_m": float(np.percentile(z, 5)), "origin_z": float(o[2]),
        })
        if lvl in (5, 9):
            ax = axes[ti, 0 if lvl == 5 else 1]
            mid = sl.shape[0] // 2
            ax.plot((np.arange(sl.shape[1]) + 0.5) * res_t - size / 2, sl[mid] - o[2], lw=0.8)
            ax.set_title(f"{name} L{lvl}: section along y through the tile centre (z relative to origin)", fontsize=8)
            ax.set_ylabel("z [m]")
            ax.grid(alpha=0.3)
axes[-1, 0].set_xlabel("y - origin [m]")
axes[-1, 1].set_xlabel("y - origin [m]")
fig.tight_layout()
fig.savefig(os.path.join(args.out, "terrain_sections.png"), dpi=150)

if args.dump_tile:
    name, lvl = args.dump_tile[0], int(args.dump_tile[1])
    c = first_col[name]
    o = origins[lvl, c]
    xs, ys = o[0] - size / 2, o[1] - size / 2
    sl, _ = tile_heights(o)
    step = max(sl.shape[0] // 25, 1)
    print(f"[audit] tile {name} L{lvl}: origin {np.round(o, 3).tolist()}, {sl.shape} cells, every {step}-th cell, mm relative to origin z; rows = x (down), cols = y (right, robots walk towards lower columns)")
    for i in range(0, sl.shape[0], step):
        print(" ".join(("   ." if np.isnan(v) else f"{int(round((v - o[2]) * 1000)):4d}") for v in sl[i, ::step]))
md = ["| type | level | max abs z [m] | p95 cell step [m] | max cell step [m] | max slope over 0.1 m [deg] | p95 height [m] | p5 height [m] |", "|---|---|---|---|---|---|---|---|"]
for r in rows_tbl:
    md.append(f"| {r['type']} | {r['level']} | {r['max_abs_z_m']:.4f} | {r['p95_dz_m']:.4f} | {r['max_dz_m']:.4f} | {r['max_slope_deg']:.1f} | {r['p95_height_m']:.4f} | {r['p5_height_m']:.4f} |")
text = "\n".join(md)
print(text)
open(os.path.join(args.out, "terrain_audit.md"), "w").write(text + "\n")
json.dump({"cfg_size": cfg.size, "rows": cfg.num_rows, "cols": cfg.num_cols, "border": cfg.border_width, "step_cap": args.step_cap or STEP_CAP, "names": names, "table": rows_tbl}, open(os.path.join(args.out, "terrain_audit.json"), "w"), indent=1)
print("[audit] wrote", os.listdir(args.out))
app.close()

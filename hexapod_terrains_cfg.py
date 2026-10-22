# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scaled rough terrain for the hexapod (decisions.md P1/P2).

The robot stands 0.10 m tall (ANYmal: 0.6 m), so the ANYmal rough terrain is scaled by about 0.17 and reduced to the
terrain types a blind 0.1 m robot can cross: random rough, slopes up to 15 degrees, and small steps / blocks below
the measured swing clearance. No stairs, boxes, pits. Tiles are 2 m, 10 rows = 10 curriculum levels.

``STEP_CAP`` is the step-height cap. It is set from the swing-clearance measurement
(``tools/measure_clearance.py``: cap = 0.8 x median clearance of the teacher the speed sweep selects). Until that
measurement exists it defaults to 0.03 m.
"""

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh

STEP_CAP = 0.007
"""Step-height cap [m] for obstacles and grid blocks = 0.8 x median swing clearance of the selected open-loop teacher.
Measured 2026-10-21 (tools/measure_clearance.py, 16 envs/setting, 10 s): teacher hip 0.5 rad / period 0.4 s, median toe
clearance 0.0088 m (p10 0.0045, p90 0.0340) -> 0.0071 m. Provisional until the learned speed sweep (part B) picks the
teacher; the current 0.35 rad / 0.8 s teacher clears ~0.017 m. Was 0.03 before the measurement."""

TILE_SIZE = 2.0
NUM_LEVELS = 10
PLATFORM_WIDTH = 0.5
"""Flat start platform at the tile centre [m]. Default choice: 25% of the tile, the same ratio as ANYmal's 2 m
platform on 8 m tiles."""
HF_BORDER = 0.04
"""Flat border of each height-field tile [m] (must be >= horizontal_scale). ANYmal uses 0.25 m on 8 m tiles."""
INCLUDE_RANDOM_GRID = True
"""Default choice: the brief lists MeshRandomGridTerrainCfg as optional; it is included so steps-up and steps-down
appear in a regular pattern as well as the random one."""


@height_field_to_mesh
def random_uniform_terrain_leveled(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Random uniform noise whose amplitude grows with the curriculum level.

    Isaac Lab's ``random_uniform_terrain`` ignores ``difficulty`` (every level has the same noise range). Here the
    upper end of the noise range is interpolated: level 0 uses (lo, lo + step), level 9 uses the full (lo, hi).
    Default choice (not in the brief): without it the random-rough column would be the same at every level and the
    per-level success bars (P4) would not mean anything for that type.
    """
    lo, hi = cfg.noise_range
    hi_d = lo + difficulty * (hi - lo)
    hi_d = max(hi_d, lo + cfg.noise_step)
    leveled = cfg.replace(noise_range=(lo, hi_d))
    return hf_terrains.random_uniform_terrain.__wrapped__(difficulty, leveled)


STAIR_TREAD = 0.12
"""Terrain v2 stair tread [m] ~ one stride of the learned gait at 0.28 m/s (measured 0.124 m median, 2026-10-21)."""
BOX_GRID_WIDTH = 0.12
"""Terrain v2 random-box grid width [m], same basis as the tread."""


def make_hexapod_terrains_cfg(
    step_cap: float = STEP_CAP, include_random_grid: bool = INCLUDE_RANDOM_GRID, version: int = 1,
    tread: float = STAIR_TREAD, grid_width: float = BOX_GRID_WIDTH, stair_platform: float = PLATFORM_WIDTH,
) -> TerrainGeneratorCfg:
    """Build the generator config.

    version 1 (2026-10-21 night): random rough, slope, inverted slope, discrete obstacles, random grid (blocks
    ±step_cap). ``step_cap`` is the obstacle / block height cap [m].
    version 2 (Shaurya, 2026-10-21 morning): random rough, slope, inverted slope, pyramid stairs, inverted pyramid
    stairs, random boxes up and down; no obstacles. Heights scale with the level so that the largest single edge at
    L9 equals ``step_cap`` (the kinematic cap): stair step height (0.005, step_cap), box heights ±step_cap/2 (box range
    (0.0025, step_cap/2), so an up-box next to a down-box is one step_cap edge). Tread / grid width from the gait
    footprint test.
    """
    if version == 2:
        return _make_v2(step_cap, tread, grid_width, stair_platform)
    rough = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.005, 0.02), noise_step=0.005, border_width=HF_BORDER
    )
    rough.function = random_uniform_terrain_leveled  # set on the instance so it is never bound as a method
    sub_terrains = {
        "random_rough": rough,
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.27), platform_width=PLATFORM_WIDTH, border_width=HF_BORDER
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.27), platform_width=PLATFORM_WIDTH, border_width=HF_BORDER
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            obstacle_height_mode="choice",  # heights in {+-h, +-h/2}: small steps up and small pits
            obstacle_height_range=(0.005, step_cap),
            obstacle_width_range=(0.05, 0.2),
            num_obstacles=40,  # default choice: ~10 per m^2 (legged_gym: 20 on 8 m tiles at 1-2 m widths)
            platform_width=PLATFORM_WIDTH,
            border_width=HF_BORDER,
        ),
    }
    if include_random_grid:
        sub_terrains["grid"] = terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.11, grid_height_range=(0.005, step_cap), platform_width=PLATFORM_WIDTH  # 0.11 not 0.10: the tile size must not be a multiple of the grid (Isaac Lab needs a non-zero border)
        )
    num_types = len(sub_terrains)
    return TerrainGeneratorCfg(
        size=(TILE_SIZE, TILE_SIZE),
        border_width=5.0,  # flat ground around the tiles. Robots walk along -y (their forward axis) and leave the
        # column-0 tiles within an episode (measured: 100% "falls" there with a 1 m border); 5 m covers 20 s at 0.25 m/s
        num_rows=NUM_LEVELS,
        num_cols=2 * num_types,  # equal proportions: two columns per type (>= 8 columns)
        horizontal_scale=0.02,
        vertical_scale=0.001,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )


def _make_v2(step_cap: float, tread: float, grid_width: float, stair_platform: float = PLATFORM_WIDTH) -> TerrainGeneratorCfg:
    rough = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.005, 0.02), noise_step=0.005, border_width=HF_BORDER
    )
    rough.function = random_uniform_terrain_leveled
    sub_terrains = {
        "random_rough": rough,
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.27), platform_width=PLATFORM_WIDTH, border_width=HF_BORDER
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.27), platform_width=PLATFORM_WIDTH, border_width=HF_BORDER
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.005, step_cap), step_width=tread, platform_width=stair_platform,
            border_width=HF_BORDER, holes=False
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.005, step_cap), step_width=tread, platform_width=stair_platform,
            border_width=HF_BORDER, holes=False
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=grid_width, grid_height_range=(0.0025, step_cap / 2), platform_width=stair_platform
        ),
    }
    return TerrainGeneratorCfg(
        size=(TILE_SIZE, TILE_SIZE),
        border_width=5.0,
        num_rows=NUM_LEVELS,
        num_cols=2 * len(sub_terrains),
        horizontal_scale=0.02,
        vertical_scale=0.001,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )


HEXAPOD_TERRAINS_CFG = make_hexapod_terrains_cfg()
"""Scaled hexapod rough terrain: 10 levels x (2 columns per type)."""


def terrain_type_names(cfg: TerrainGeneratorCfg) -> list[str]:
    """Sub-terrain name of each column, with the same rule ``TerrainGenerator._generate_curriculum_terrains`` uses."""
    names = list(cfg.sub_terrains.keys())
    proportions = np.array([sub.proportion for sub in cfg.sub_terrains.values()], dtype=float)
    proportions /= proportions.sum()
    out = []
    for col in range(cfg.num_cols):
        out.append(names[int(np.min(np.where(col / cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))])
    return out

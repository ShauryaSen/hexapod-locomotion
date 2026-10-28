# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from .hexapod_robot_cfg import HEXAPOD_CFG  # isort: skip
from .hexapod_terrains_cfg import HEXAPOD_TERRAINS_CFG  # isort: skip


@configclass
class DomainRandCfg:
    """Per-env domain randomization, applied by the env itself (not the event manager) so the true values are known
    and can be exposed as the privileged observation ``(delta_mass_kg, mu_static, motor_strength)``.

    Ranges follow decisions.md D1-D3 (status *lean*): friction 0.5-1.25, mass +-0.75 kg, motor strength 0.8-1.2.
    """

    enabled: bool = True
    """Sample new values for every env at every episode reset (and at startup). False = nominal values."""

    friction_static_range: tuple[float, float] = (0.5, 1.25)
    """Static friction of every collision shape of the robot, U(lo, hi) per env. The ground has mu = 1.0 and the
    combine mode is multiply, so this is the effective robot-ground friction."""
    friction_dynamic_ratio: float = 0.75
    """Dynamic friction = static * this ratio (default choice, not in decisions.md: the old fixed material used
    0.8 / 0.6 = 0.75)."""
    restitution: float = 0.0

    mass_add_range_kg: tuple[float, float] = (-0.75, 0.75)
    """Added to the mass of the ``body`` link, U(lo, hi) per env; inertia scaled with the mass ratio."""

    motor_strength_range: tuple[float, float] = (0.8, 1.2)
    """Motor strength s ~ U(lo, hi) per env: stiffness, damping and the effort limit of all joints are scaled by s,
    so the PD torque is exactly s times the nominal one (decisions.md D2)."""


@configclass
class HexapodFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 8
    # video camera next to env 0 (the default eye is 7.5 m away; the robot is ~0.1 m tall)
    viewer: ViewerCfg = ViewerCfg(eye=(0.45, -0.45, 0.22), lookat=(0.0, 0.0, 0.1), origin_type="asset_root", asset_name="robot")
    action_scale = 0.35  # club value; measured fewer early falls than ANYmal's 0.5 (3.5% vs 4.4% per step)
    action_space = 18
    observation_space = 66
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    filter_env_collisions: bool = True
    """Call scene.filter_collisions on GPU too (see HexapodEnv._setup_scene)."""

    # events: none. Friction / mass / motor-strength randomization is done by the env (see ``dr``) so that the
    # sampled values are known per env (privileged observation) and can be set at runtime.
    events = None

    # domain randomization (decisions.md D1-D3)
    dr: DomainRandCfg = DomainRandCfg()

    # robot
    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # perceptive variant: height scanner appended to the observation (187 values). Off = blind (decisions.md P1).
    perceptive: bool = False
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # commands: at each reset, x/y velocity and yaw rate are sampled uniformly in [-max, max]
    command_lin_vel_max: float = 0.3
    """Largest commanded x or y velocity [m/s]."""
    command_yaw_rate_max: float = 0.3
    """Largest commanded yaw rate [rad/s]."""
    # Forward/backward-only commands: |v| ~ U(command_lin_vel_min, command_lin_vel_max) along body -y/+y (the
    # club's forward axis is -y), x = 0. Used with the behavior-cloned tripod, whose teacher only walks forward/back.
    command_forward_only: bool = False
    command_lin_vel_min: float = 0.0
    command_backward_fraction: float = 0.5
    """Fraction of forward-only episodes that walk backward (+y). 0.5 was the behaviour before this field existed."""

    # Command curriculum (speed sweep, decisions.md P9): command_lin_vel_max starts at ``command_curriculum_start``
    # and is raised by ``command_curriculum_step`` whenever, over the last ``command_curriculum_window_episodes``
    # episodes whose command was in the top bucket [max - step, max], the success rate (success_eval definition)
    # is >= ``command_curriculum_success``. Capped at ``command_curriculum_max``. Logged as Curriculum/command_lin_vel_max.
    command_curriculum: bool = False
    command_curriculum_start: float = 0.15
    command_curriculum_step: float = 0.05
    command_curriculum_max: float = 0.4
    command_curriculum_success: float = 0.8
    command_curriculum_window_episodes: int = 256
    """Default choice (not in the brief, which says "last 20 iterations"): 20 iterations x 24 steps is 9.6 s, less
    than one 20 s episode, so the window is counted in finished top-bucket episodes instead."""

    # Gait clock: when enabled, [sin, cos] of a phase with period gait_period_s is appended to the observation
    # (observation_space += 2) and advances every step. The behavior-cloned policy needs it: the teacher gait is a
    # function of this phase, and a memoryless MLP cannot keep time on its own from a standing start.
    gait_clock_enabled: bool = False
    gait_period_s: float = 0.8

    # Reset spreading: on a full reset, episode_length_buf is randomized so training envs do not all reset
    # together. The eval suite turns this off so every env runs a full episode.
    spread_resets: bool = True

    # velocity-tracking tolerance: reward = exp(-error^2 / sigma^2) with sigma = ratio * max command.
    # ANYmal uses commands up to 1.0 with sigma = 0.5 (the hard-coded 0.25 = 0.5^2), i.e. ratio 0.5.
    tracking_sigma_ratio: float = 0.5

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    # Measured under a random policy (diag_reward_terms.py): the hexapod's summed torque^2 is ~500x
    # smaller than ANYmal's (1.6 N·m motors vs 80 N·m) and its summed joint accel^2 ~100x larger
    # (light legs). The scales are shifted by those ratios so each penalty keeps ANYmal's weight.
    joint_torque_reward_scale = -1.25e-2  # ANYmal: -2.5e-5
    joint_accel_reward_scale = -2.5e-9  # ANYmal: -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    # Swing clearance (added 2026-10-22 for stairs; 0 = off, the behaviour before this field existed). For every
    # foot not in contact while the robot is commanded to move, reward min(toe height above the mean height of the
    # feet in contact, target) / target. Blind-policy proxy for "lift the feet over an unseen riser" (legged_gym-style
    # feet-clearance term). Default target 0.06 m: just above the 0.052 m stair cap.
    swing_clearance_reward_scale = 0.0
    swing_clearance_target = 0.06
    # A touchdown pays (swing time - threshold): below it, steps are penalized. ANYmal uses 0.5 s; for this
    # robot that left standing (0 reward) strictly better than any first short step, so it never stepped.
    feet_air_time_threshold = 0.2  # [s]
    # Tripod rhythm (added 2026-10-28, "walk normally"; 0 = off, the behaviour before this field existed). Per step,
    # the fraction of the six legs whose contact state (tibia contact force > 1 N) equals the gait clock's tripod
    # schedule (ScriptedTripod convention: lf, rm, lr in stance for clock phase in [0, pi), rf, lm, rr in
    # [pi, 2 pi)); paid only while the robot is commanded to move (same gate as the air-time reward). Needs the
    # gait clock.
    tripod_rhythm_reward_scale = 0.0
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0
    # One-off penalty when the body hits the ground. The hexapod falls ~4% of steps under random
    # actions (ANYmal: 0%), so without this, ending the episode early is the cheapest policy.
    # Same weight as Isaac Lab's Cassie rough config (-200 * step_dt = -4 per termination).
    termination_reward_scale = -200.0

    # Success (decisions.md P3), logged as Metrics/success_eval: the episode reached time-out (no fall) AND
    # | episode-mean speed along the command direction - |command| | < success_speed_tol_ratio * |command|.
    success_speed_tol_ratio: float = 0.25
    success_min_cmd_ratio: float = 0.1
    """Default choice: for commands below this fraction of command_lin_vel_max (the same "moving" threshold the
    air-time reward uses) the tolerance is computed against that floor instead of |command|, so a zero command
    does not make the test degenerate."""

    # Old success definition kept for comparison, logged as Metrics/success_rate_old: per-env binary check that
    # the episode-mean XY velocity error norm and yaw error stayed below ratio * max command. It counts falls as
    # successes and scales with the max command (that is why P3 replaced it).
    vel_xy_success_ratio: float = 0.5
    vel_yaw_success_ratio: float = 0.4

    # Terrain curriculum (decisions.md P2). Only active when the terrain is a generator with curriculum rows.
    terrain_curriculum: bool = False
    curriculum_move_down_speed_ratio: float = 0.5
    """move_down when the episode-mean speed along the command is below this fraction of |command| (or a fall)."""


@configclass
class HexapodRoughEnvCfg(HexapodFlatEnvCfg):
    """Blind rough-terrain variant (decisions.md P1/P2): scaled terrain tiles with a 10-level curriculum, everyone
    starts at level 0, no height scanner (observation 66 + clock)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HEXAPOD_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # env origins come from the terrain tiles (2 m); env_spacing only places the cloned prims before the first reset
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    terrain_curriculum: bool = True
    perceptive: bool = False
    step_cap: float | None = None
    """Obstacle / grid block height cap [m] for this run; None keeps hexapod_terrains_cfg.STEP_CAP. The env rebuilds
    the generator config with make_hexapod_terrains_cfg(step_cap, version=terrain_version) before the scene is created."""
    terrain_version: int = 1
    terrain_stair_tread: float | None = None
    """Terrain v2 stair tread [m]; None = hexapod_terrains_cfg.STAIR_TREAD (0.12). Added for the stairs doability study."""
    terrain_tile_size: float | None = None
    """Terrain v2 tile size [m]; None = TILE_SIZE (2.0). For the platform-width eval (2026-10-22), kept with the
    platform so the stair count per side is unchanged: Isaac Lab steps = (tile - 2*0.04 - platform) // (2*tread) + 1."""
    terrain_stair_platform: float | None = None
    """Terrain v2 platform width [m] of the stair (apex / pit) and box tiles; None = PLATFORM_WIDTH (0.5)."""
    """1 = the 2026-10-21 night set (rough, slopes, obstacles, grid); 2 = Shaurya's set (rough, slopes, stairs both
    ways, boxes up and down). With version 2 the env always rebuilds the generator config."""

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

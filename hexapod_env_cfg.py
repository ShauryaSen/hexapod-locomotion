# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from .hexapod_robot_cfg import HEXAPOD_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )


@configclass
class HexapodFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 8
    # video camera close to env 0 (default eye is 7.5 m away and the robot is ~0.1 m tall)
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

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # commands: at each reset, x/y velocity and yaw rate are sampled uniformly in [-max, max]
    command_lin_vel_max: float = 0.3
    """Largest commanded x or y velocity [m/s]."""
    command_yaw_rate_max: float = 0.3
    """Largest commanded yaw rate [rad/s]."""
    # exp6+: forward-only commands. If True the command is +-U(command_lin_vel_min, command_lin_vel_max) along
    # body -y/+y (the club's forward axis is -y), x = 0. Otherwise x and y are uniform in [-max, max] as before.
    command_forward_only: bool = False
    command_lin_vel_min: float = 0.0

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
    # A touchdown pays (swing time - threshold): below it, steps are penalized. ANYmal uses 0.5 s; for this
    # robot that left standing (0 reward) strictly better than any first short step, so it never stepped.
    feet_air_time_threshold = 0.2  # [s]
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0
    # One-off penalty when the body hits the ground. The hexapod falls ~4% of steps under random
    # actions (ANYmal: 0%), so without this, ending the episode early is the cheapest policy.
    # Same weight as Isaac Lab's Cassie rough config (-200 * step_dt = -4 per termination).
    termination_reward_scale = -200.0
    # Overnight 2026-10-13 (exp4+): tripod gait clock. With gait_reward_scale != 0 the policy observes
    # [sin, cos] of a clock with period gait_period_s (observation_space += 2) and earns
    # scale * (fraction of feet matching their phase - 0.5): tripod A (lf, rm, lr) swings in the first
    # half of the cycle, tripod B in the second. Standing or holding feet up scores 0; a tripod 0.5/s.
    gait_period_s: float = 0.8
    gait_reward_scale: float = 0.0  # the env adds 2 to observation_space itself when this is non-zero
    # exp5+: what counts as "swinging" / "in stance" for the clock reward. With clearance 0 (exp4) a foot
    # counts as swinging whenever its contact force is below gait_stance_force_n, which a policy can game by
    # unloading feet in place. With clearance > 0 the toe must be that far above its standing height.
    gait_stance_force_n: float = 1.0
    gait_swing_clearance_m: float = 0.0
    gait_smooth: bool = False  # exp6+: partial credit = clamp(lift/clearance, 0, 1) in swing, clamp(force/stance_n, 0, 1) in stance
    # exp7+: stance-foot velocity tracking. Feet in their stance half (and loaded) are rewarded for moving through
    # the body frame at -command (what a planted foot does when the body advances): scale * mean exp(-|v_foot_b + c|^2 / sigma^2).
    stance_foot_vel_reward_scale: float = 0.0
    stance_foot_vel_sigma: float = 0.1
    # exp8+: imitation of the Phase 0 scripted tripod. Reference joint deltas are a function of the clock phase and
    # the command (sweep sign from the sign of cmd_y, hip amplitude = clamp(|cmd_y| / 0.45, 0, ref_hip_max)):
    # reward = scale * exp(-mean_j (q_j - q_ref_j)^2 / ref_gait_sigma^2). Phase 0 measured: hip 0.2 rad, T 0.8 s -> ~0.09-0.1 m/s.
    ref_gait_reward_scale: float = 0.0
    # exp10+: residual control. Joint targets = default + reference tripod deltas + action_scale * action, so the
    # robot walks the scripted gait at iteration 0 and the policy learns corrections (tracking, balance).
    ref_gait_residual: bool = False
    ref_gait_sigma: float = 0.15
    ref_hip_max: float = 0.35
    ref_lift_femur: float = 0.35
    ref_lift_tibia: float = -0.30
    # 2026-10-13 pure-RL attempt (rl1). All default-off; no scripted gait reaches the policy's actions.
    # (1) linear progress: scale * clip(v_xy . cmd_dir, 0, |cmd|) / command_lin_vel_max. The exp tracking term pays
    # ~nothing until the error is within sigma, so a slow shuffle scored like standing; this pays for every cm/s.
    progress_reward_scale: float = 0.0
    # (2) reference-state initialization (DeepMimic-style): with probability rsi_prob an episode starts in a random
    # phase of the scripted tripod *pose* (from _ref_joint_delta) with root velocity = command. Reset only.
    rsi_prob: float = 0.0
    # (3) command curriculum: sampled |command| is capped at a running max that starts at command_curriculum_start and
    # grows by command_curriculum_step (up to command_lin_vel_max) whenever the EMA of episode-mean err_xy drops below
    # command_curriculum_err_ratio * current max. The tracking sigma follows the current max. Metrics/cmd_max_cur.
    command_curriculum: bool = False
    command_curriculum_start: float = 0.06
    command_curriculum_step: float = 0.02
    command_curriculum_err_ratio: float = 0.5
    # (4) penalty curriculum: torque, joint-accel and action-rate penalties are multiplied by
    # min + (1 - min) * clip(env_step / penalty_curriculum_steps, 0, 1); 0 = off. Metrics/penalty_factor.
    penalty_curriculum_steps: int = 0
    penalty_curriculum_min: float = 0.1
    # rl2: low-pass the body xy velocity (EMA, time constant vel_filter_tau_s) before the tracking, progress and
    # curriculum-error terms. rl1 measured 0.08/s of "progress" from exploration jitter alone (deterministic rollout:
    # 0.0004 m/s); zero-mean jitter averages out over ~a stride, real travel does not. 0 = off (raw velocity).
    vel_filter_tau_s: float = 0.0
    # rl4: stand-still termination. rl3 measured a "ratchet": the stochastic policy earns 0.3-0.4/s of progress from
    # noise-driven drift while the mean action stands (deterministic rollout 0.0005 m/s), so no action direction is
    # preferred and the mean never moves. If a commanded robot's (filtered) speed along the command stays below
    # standstill_speed_ratio * |cmd| for standstill_terminate_s seconds, the episode ends like a fall (same penalty).
    standstill_terminate_s: float = 0.0  # 0 = off
    standstill_speed_ratio: float = 0.5
    # rl6: per-leg step reward on the actuated hip joint. Desired coxa velocity direction = (sign that moves the body
    # along the command) x (-1 while the foot is loaded > gait_stance_force_n, +1 while its toe is lifted >
    # gait_swing_clearance_m above standing, 0 otherwise). reward = mean_legs clamp(dq_coxa x dir / step_ref_vel, -1, 1),
    # only when commanded. Standing scores 0 and symmetric random twitches average 0 (the rl3 ratchet cannot earn it).
    step_reward_scale: float = 0.0
    step_ref_vel: float = 1.0  # [rad/s] coxa speed that earns the full reward
    # dm1 (DeepMimic-style, 2026-10-13 afternoon). The scripted tripod is used in training only: as a reward target
    # and for reset states. Nothing scripted reaches the actions.
    gait_clock_enabled: bool = False  # phase [sin, cos] in the observation and advancing, even with gait_reward_scale = 0
    rsi_joint_vel: bool = False  # RSI also sets joint velocities = d(reference)/dt, so a started stride carries itself
    # imitation weight anneal: ref_gait weight x clamp(1 - (env_step - start) / steps, 0, 1); steps = 0 disables
    ref_gait_anneal_start_steps: int = 0
    ref_gait_anneal_steps: int = 0
    gait_toe_standing_z_m: float = 0.0078  # measured 2026-10-13 00:50 (diag_gait_clock_smoke.py): toe height in the default stance

    # success criteria — episode success_rate = per-env binary check that the *episode-mean*
    # error stayed below both thresholds, mean-reduced across resetting envs.
    # Thresholds are ratio * max command; ANYmal's 0.5 m/s and 0.4 rad/s at max 1.0 give these ratios.
    vel_xy_success_ratio: float = 0.5
    """Threshold on the per-episode mean XY velocity error norm, as a fraction of command_lin_vel_max."""
    vel_yaw_success_ratio: float = 0.4
    """Threshold on the per-episode mean yaw velocity error, as a fraction of command_yaw_rate_max."""


@configclass
class HexapodRoughEnvCfg(HexapodFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
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

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

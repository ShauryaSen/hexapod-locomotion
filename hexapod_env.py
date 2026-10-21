# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .hexapod_env_cfg import HexapodFlatEnvCfg, HexapodRoughEnvCfg
from .hexapod_terrains_cfg import terrain_type_names

BASE_OBS_DIM = 66
HEIGHT_SCAN_DIM = 187  # 1.6 m x 1.0 m grid at 0.1 m = 17 x 11 rays
PRIVILEGED_DIM = 3  # (delta_mass_kg, mu_static, motor_strength)


class HexapodEnv(DirectRLEnv):
    cfg: HexapodFlatEnvCfg | HexapodRoughEnvCfg

    def __init__(self, cfg: HexapodFlatEnvCfg | HexapodRoughEnvCfg, render_mode: str | None = None, **kwargs):
        # observation size is derived here because Hydra overrides land after cfg init
        cfg.observation_space = (
            BASE_OBS_DIM + (HEIGHT_SCAN_DIM if cfg.perceptive else 0) + (2 if cfg.gait_clock_enabled else 0)
        )
        super().__init__(cfg, render_mode, **kwargs)
        # second observation group: the true DR values (RMA teacher input). Plain PPO maps only "policy" to the
        # actor and critic (rsl_rl resolve_obs_groups), so this group is ignored unless obs_groups lists it.
        self.single_observation_space["privileged"] = gym.spaces.Box(-math.inf, math.inf, (PRIVILEGED_DIM,))

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # command curriculum state (speed sweep, part B)
        self.command_lin_vel_max = (
            float(self.cfg.command_curriculum_start) if self.cfg.command_curriculum else float(self.cfg.command_lin_vel_max)
        )
        self._cc_window_n = 0
        self._cc_window_success = 0

        # Per-episode accumulators. Sums over actual episode steps; means are finalized at reset, so logged values
        # describe the just-ended episode rather than the post-reset instant.
        n = self.num_envs
        self._error_xy_sum = torch.zeros(n, device=self.device)
        self._error_yaw_sum = torch.zeros(n, device=self.device)
        self._v_along_cmd_sum = torch.zeros(n, device=self.device)  # speed along the command direction
        self._step_count = torch.zeros(n, device=self.device)
        self._root_z_sum = torch.zeros(n, device=self.device)  # logged as Metrics/root_height
        self._speed_xy_sum = torch.zeros(n, device=self.device)
        self._swing_sum = torch.zeros(n, device=self.device)
        self._torque_sat_sum = torch.zeros(n, device=self.device)
        # gait clock phase [rad], one per env
        self._gait_phase = torch.zeros(n, device=self.device)
        # stats of the last finished episode of every env (filled at reset; read by tools/eval_suite.py)
        self.episode_stats = {
            key: torch.zeros(n, device=self.device)
            for key in ("success", "fell", "speed_err", "mean_v_along_cmd", "cmd_speed", "length_s", "terrain_level")
        }

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "termination",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_sensors("body")
        self._feet_ids, _ = self._contact_sensor.find_sensors(".*_tibia")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_sensors(".*_(coxa|femur)")  # measured: a trained policy sat on its coxas (37 N) and nothing penalized it

        # terrain curriculum bookkeeping
        self._has_terrain_levels = getattr(self._terrain, "terrain_origins", None) is not None
        self.terrain_type_names = (
            terrain_type_names(self.cfg.terrain.terrain_generator) if self._has_terrain_levels else ["flat"]
        )

        # domain randomization: nominal values, current values, setters
        self._setup_dr()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if self.cfg.perceptive:
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # Filter collisions between envs explicitly (the Isaac Lab template only does this on CPU). Measured
        # 2026-10-21 on Isaac Lab 3.0 / GPU: with a generator terrain, envs that share a tile origin spawned inside
        # each other (10+ kN body forces, instant "falls"); the implicit PhysX env-id separation did not prevent it.
        if self.device == "cpu" or self.cfg.filter_env_collisions:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------------------------------------------------
    # Domain randomization (decisions.md D1-D4) and runtime setters for the deployment schedule
    # ------------------------------------------------------------------------------------------------------------

    def _setup_dr(self):
        dev = self.device
        n = self.num_envs
        self._body_id = self._robot.find_bodies("body")[0]
        self._nominal_body_mass = self._robot.data.body_mass.torch[:, self._body_id].clone()  # (N, 1)
        self._nominal_body_inertia = self._robot.data.body_inertia.torch[:, self._body_id].clone()  # (N, 1, 9)
        self._nominal_stiffness = self._robot.data.joint_stiffness.torch.clone()  # (N, J)
        self._nominal_damping = self._robot.data.joint_damping.torch.clone()
        self._nominal_effort_limit = self._robot.data.joint_effort_limits.torch.clone()
        self._actuator = self._robot.actuators["legs"]
        self._all_joint_ids = torch.arange(self._robot.num_joints, device=dev, dtype=torch.int32)  # warp wants int32
        # current values per env (the privileged observation)
        self.dr_delta_mass = torch.zeros(n, device=dev)
        self.dr_friction = torch.ones(n, device=dev) * float(self.cfg.sim.physics_material.static_friction)
        self.dr_motor_strength = torch.ones(n, device=dev)
        self.dr_hold = torch.zeros(n, dtype=torch.bool, device=dev)
        """Envs whose values were set by a runtime setter with hold=True; the reset resampling skips them."""
        # material buffer: friction is per collision shape; the robot's shapes all get the same value
        materials = wp.to_torch(self._robot.root_view.get_material_properties())
        self._material_shape = tuple(materials.shape)  # (N, num_shapes, 3)
        if self.cfg.dr.enabled:
            self.resample_dr(self._robot._ALL_INDICES)

    def _env_ids_tensor(self, env_ids) -> torch.Tensor:
        if isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids)
        return torch.as_tensor(env_ids, device=self.device).long()

    def resample_dr(self, env_ids):
        """Sample new friction / mass / motor strength for ``env_ids`` from the cfg ranges (skips held envs)."""
        env_ids = self._env_ids_tensor(env_ids)
        env_ids = env_ids[~self.dr_hold[env_ids]]
        if len(env_ids) == 0:
            return
        cfg = self.cfg.dr
        u = lambda lo, hi: torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)
        self.set_friction(env_ids, u(*cfg.friction_static_range), hold=False)
        self.set_payload(env_ids, u(*cfg.mass_add_range_kg), hold=False)
        self.set_motor_strength(env_ids, u(*cfg.motor_strength_range), hold=False)

    def set_friction(self, env_ids, mu_static, hold: bool = True):
        """Set the static friction of every collision shape of the robot in ``env_ids`` (dynamic = ratio x static).
        Takes effect at the next physics step (PhysX material update). ``hold=True`` protects the envs from reset
        resampling until ``release_dr`` is called."""
        env_ids = self._env_ids_tensor(env_ids)
        mu = torch.as_tensor(mu_static, device=self.device, dtype=torch.float32).expand(len(env_ids)).clone()
        self.dr_friction[env_ids] = mu
        self.dr_hold[env_ids] |= hold
        materials = wp.to_torch(self._robot.root_view.get_material_properties())  # CPU buffer (N, S, 3)
        ids_cpu = env_ids.cpu().to(torch.int32)
        mu_cpu = mu.cpu()
        materials[ids_cpu, :, 0] = mu_cpu[:, None]
        materials[ids_cpu, :, 1] = mu_cpu[:, None] * self.cfg.dr.friction_dynamic_ratio
        materials[ids_cpu, :, 2] = self.cfg.dr.restitution
        self._robot.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(ids_cpu, dtype=wp.int32)
        )

    def set_payload(self, env_ids, delta_mass_kg, hold: bool = True):
        """Add ``delta_mass_kg`` to the nominal mass of the ``body`` link (inertia scaled with the mass ratio)."""
        env_ids = self._env_ids_tensor(env_ids)
        dm = torch.as_tensor(delta_mass_kg, device=self.device, dtype=torch.float32).expand(len(env_ids)).clone()
        self.dr_delta_mass[env_ids] = dm
        self.dr_hold[env_ids] |= hold
        nominal = self._nominal_body_mass[env_ids]  # (n, 1)
        masses = (nominal + dm[:, None]).clamp_min(1e-3)
        ratio = masses / nominal
        body_ids = torch.as_tensor(self._body_id, device=self.device, dtype=torch.int32)
        ids32 = env_ids.to(torch.int32)
        self._robot.set_masses_index(masses=masses.contiguous(), body_ids=body_ids, env_ids=ids32)
        self._robot.set_inertias_index(
            inertias=(self._nominal_body_inertia[env_ids] * ratio[..., None]).contiguous(), body_ids=body_ids, env_ids=ids32
        )

    def set_motor_strength(self, env_ids, strength, hold: bool = True):
        """Scale stiffness, damping and the effort limit of all joints by ``strength`` (torque = s x nominal)."""
        env_ids = self._env_ids_tensor(env_ids)
        s = torch.as_tensor(strength, device=self.device, dtype=torch.float32).expand(len(env_ids)).clone()
        self.dr_motor_strength[env_ids] = s
        self.dr_hold[env_ids] |= hold
        kp = (self._nominal_stiffness[env_ids] * s[:, None]).contiguous()
        kd = (self._nominal_damping[env_ids] * s[:, None]).contiguous()
        lim = (self._nominal_effort_limit[env_ids] * s[:, None]).contiguous()
        ids32 = env_ids.to(torch.int32)
        self._robot.write_joint_stiffness_to_sim_index(stiffness=kp, joint_ids=self._all_joint_ids, env_ids=ids32)
        self._robot.write_joint_damping_to_sim_index(damping=kd, joint_ids=self._all_joint_ids, env_ids=ids32)
        self._robot.write_joint_effort_limit_to_sim_index(limits=lim, joint_ids=self._all_joint_ids, env_ids=ids32)
        # the implicit actuator's torque estimate (applied_torque, used by the torque reward and Metrics/torque_sat)
        # uses its own copies of the gains and the limit
        self._actuator.stiffness[env_ids] = kp
        self._actuator.damping[env_ids] = kd
        self._actuator.effort_limit[env_ids] = lim

    def release_dr(self, env_ids):
        """Let reset resampling touch ``env_ids`` again."""
        self.dr_hold[self._env_ids_tensor(env_ids)] = False

    def set_nominal_dr(self, env_ids):
        """Nominal robot (no payload, friction of the sim material, strength 1) and release the hold."""
        env_ids = self._env_ids_tensor(env_ids)
        self.set_friction(env_ids, float(self.cfg.sim.physics_material.static_friction), hold=False)
        self.set_payload(env_ids, 0.0, hold=False)
        self.set_motor_strength(env_ids, 1.0, hold=False)
        self.dr_hold[env_ids] = False

    def privileged_obs(self) -> torch.Tensor:
        return torch.stack([self.dr_delta_mass, self.dr_friction, self.dr_motor_strength], dim=1)

    # ------------------------------------------------------------------------------------------------------------
    # Terrain helpers (eval suite)
    # ------------------------------------------------------------------------------------------------------------

    def set_terrain_cells(self, env_ids, levels, types):
        """Pin ``env_ids`` to terrain (level, type) cells; the next reset places them there."""
        if not self._has_terrain_levels:
            return
        env_ids = self._env_ids_tensor(env_ids)
        self._terrain.terrain_levels[env_ids] = torch.as_tensor(levels, device=self.device).long().expand(len(env_ids))
        self._terrain.terrain_types[env_ids] = torch.as_tensor(types, device=self.device).long().expand(len(env_ids))
        self._terrain.env_origins[env_ids] = self._terrain.terrain_origins[
            self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]
        ]

    # ------------------------------------------------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        if self.cfg.gait_clock_enabled:
            self._gait_phase = torch.remainder(self._gait_phase + 2 * math.pi * self.step_dt / self.cfg.gait_period_s, 2 * math.pi)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if self.cfg.perceptive:
            height_data = (
                self._height_scanner.data.pos_w.torch[:, 2].unsqueeze(1)
                - self._height_scanner.data.ray_hits_w.torch[..., 2]
                - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b.torch,
                    self._robot.data.root_ang_vel_b.torch,
                    self._robot.data.projected_gravity_b.torch,
                    self._commands,
                    self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch,
                    self._robot.data.joint_vel.torch,
                    height_data,
                    self._actions,
                    torch.stack([torch.sin(self._gait_phase), torch.cos(self._gait_phase)], dim=1) if self.cfg.gait_clock_enabled else None,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs, "privileged": self.privileged_obs()}

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b.torch[:, :2]), dim=1
        )
        # tolerance sigma scales with the command range, so standing still is never "close enough"
        lin_vel_sigma = self.cfg.tracking_sigma_ratio * self.command_lin_vel_max
        lin_vel_error_mapped = torch.exp(-lin_vel_error / lin_vel_sigma**2)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b.torch[:, 2])
        yaw_rate_sigma = self.cfg.tracking_sigma_ratio * self.cfg.command_yaw_rate_max
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / yaw_rate_sigma**2)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b.torch[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b.torch[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque.torch), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc.torch), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt).torch[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time.torch[:, self._feet_ids]
        # only reward stepping when the robot is asked to move (ANYmal: > 0.1 m/s at a 1.0 m/s command max)
        air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1) * (
            torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1 * self.command_lin_vel_max
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        is_contact = (
            torch.max(torch.linalg.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b.torch[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            # _get_dones runs before _get_rewards, so reset_terminated already holds this step's deaths
            "termination": self.reset_terminated.float() * self.cfg.termination_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Accumulate per-episode sums; the per-episode metrics are finalized in _reset_idx.
        v_xy = self._robot.data.root_lin_vel_b.torch[:, :2]
        cmd_xy = self._commands[:, :2]
        cmd_speed = torch.linalg.norm(cmd_xy, dim=-1)
        cmd_dir = cmd_xy / cmd_speed.clamp_min(1e-6).unsqueeze(1)
        self._v_along_cmd_sum += torch.sum(v_xy * cmd_dir, dim=-1)
        self._error_xy_sum += torch.linalg.norm(cmd_xy - v_xy, dim=-1)
        self._error_yaw_sum += torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b.torch[:, 2])
        self._step_count += 1.0
        self._root_z_sum += self._robot.data.root_pos_w.torch[:, 2]
        self._speed_xy_sum += torch.linalg.norm(v_xy, dim=-1)
        feet_forces = torch.linalg.norm(self._contact_sensor.data.net_forces_w.torch[:, self._feet_ids], dim=-1)
        self._swing_sum += torch.any(feet_forces < 1.0, dim=1).float()
        self._torque_sat_sum += (
            self._robot.data.applied_torque.torch.abs() > 0.95 * self._actuator.effort_limit
        ).float().mean(dim=1)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        died = torch.any(
            torch.max(torch.linalg.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        return died, time_out

    # ------------------------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------------------------

    def _episode_outcome(self, env_ids: torch.Tensor):
        """Per-env outcome of the episode that just ended (decisions.md P3). Envs with no steps count as not run."""
        steps = self._step_count[env_ids]
        denom = steps.clamp_min(1.0)
        mean_v = self._v_along_cmd_sum[env_ids] / denom
        cmd_speed = torch.linalg.norm(self._commands[env_ids, :2], dim=-1)
        tol_base = torch.maximum(cmd_speed, torch.full_like(cmd_speed, self.cfg.success_min_cmd_ratio * self.command_lin_vel_max))
        speed_err = torch.abs(mean_v - cmd_speed)
        fell = self.reset_terminated[env_ids]
        timed_out = self.reset_time_outs[env_ids] & ~fell
        success = timed_out & (speed_err < self.cfg.success_speed_tol_ratio * tol_base) & (steps > 0)
        slow = mean_v < self.cfg.curriculum_move_down_speed_ratio * cmd_speed
        return success, fell, slow, speed_err, mean_v, cmd_speed, steps

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        env_ids = env_ids.long()
        # ---- outcome of the ending episodes (before anything is overwritten)
        success, fell, slow, speed_err, mean_v, cmd_speed, steps = self._episode_outcome(env_ids)
        ran = steps > 0
        self.episode_stats["success"][env_ids] = success.float()
        self.episode_stats["fell"][env_ids] = fell.float()
        self.episode_stats["speed_err"][env_ids] = speed_err
        self.episode_stats["mean_v_along_cmd"][env_ids] = mean_v
        self.episode_stats["cmd_speed"][env_ids] = cmd_speed
        self.episode_stats["length_s"][env_ids] = steps * self.step_dt
        if self._has_terrain_levels:
            self.episode_stats["terrain_level"][env_ids] = self._terrain.terrain_levels[env_ids].float()
        # ---- terrain curriculum (decisions.md P2): up on success, down on a fall or < 50% of the command speed
        if self.cfg.terrain_curriculum and self._has_terrain_levels:
            move_up = success & ran
            move_down = (fell | slow) & ran & ~move_up
            self._terrain.update_env_origins(env_ids, move_up, move_down)
        # ---- command curriculum (speed sweep part B): finished episodes in the top command bucket
        if self.cfg.command_curriculum:
            top = ran & (cmd_speed >= self.command_lin_vel_max - self.cfg.command_curriculum_step - 1e-6)
            self._cc_window_n += int(top.sum().item())
            self._cc_window_success += int((top & success).sum().item())
            if self._cc_window_n >= self.cfg.command_curriculum_window_episodes:
                if (
                    self._cc_window_success / self._cc_window_n >= self.cfg.command_curriculum_success
                    and self.command_lin_vel_max < self.cfg.command_curriculum_max - 1e-6
                ):
                    self.command_lin_vel_max = min(
                        self.command_lin_vel_max + self.cfg.command_curriculum_step, self.cfg.command_curriculum_max
                    )
                self._cc_window_n = 0
                self._cc_window_success = 0

        ids32 = env_ids.to(torch.int32)  # sim writes go through warp kernels, which want int32 indices
        self._robot.reset(ids32)
        super()._reset_idx(ids32)
        if len(env_ids) == self.num_envs and self.cfg.spread_resets:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        # domain randomization: new values for the new episode
        if self.cfg.dr.enabled:
            self.resample_dr(env_ids)
        # Sample new commands
        lin_max, yaw_max = self.command_lin_vel_max, self.cfg.command_yaw_rate_max
        if self.cfg.command_forward_only:
            mag = torch.zeros(len(env_ids), device=self.device).uniform_(self.cfg.command_lin_vel_min, lin_max)
            sign = torch.where(torch.rand(len(env_ids), device=self.device) < self.cfg.command_backward_fraction, 1.0, -1.0)
            self._commands[env_ids, 0] = 0.0
            self._commands[env_ids, 1] = mag * sign
        else:
            self._commands[env_ids, :2] = torch.zeros_like(self._commands[env_ids, :2]).uniform_(-lin_max, lin_max)
        self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 2]).uniform_(-yaw_max, yaw_max)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids]
        joint_vel = self._robot.data.default_joint_vel.torch[env_ids]
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]
        default_root_pose[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=ids32)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=ids32)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids32)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=ids32)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_xy = self._error_xy_sum[env_ids] / denom
        mean_error_yaw = self._error_yaw_sum[env_ids] / denom
        extras["Metrics/error_vel_xy"] = mean_error_xy.mean().item()
        extras["Metrics/error_vel_yaw"] = mean_error_yaw.mean().item()
        extras["Metrics/speed_err_along_cmd"] = speed_err.mean().item()
        extras["Metrics/root_height"] = (self._root_z_sum[env_ids] / denom).mean().item()
        extras["Metrics/speed_xy"] = (self._speed_xy_sum[env_ids] / denom).mean().item()
        extras["Metrics/swing_frac"] = (self._swing_sum[env_ids] / denom).mean().item()
        extras["Metrics/torque_sat"] = (self._torque_sat_sum[env_ids] / denom).mean().item()
        # success, decisions.md P3 (time-out without a fall AND speed error < 25% of the command)
        extras["Metrics/success_eval"] = success.float().mean().item()
        # falls per episode-second over the ending episodes
        extras["Metrics/falls_per_episode_s"] = (fell.float().sum() / (steps.sum() * self.step_dt).clamp_min(1e-6)).item()
        # old definition (counts falls as successes; thresholds scale with the max command)
        extras["Metrics/success_rate_old"] = (
            (
                (mean_error_xy < self.cfg.vel_xy_success_ratio * self.command_lin_vel_max)
                & (mean_error_yaw < self.cfg.vel_yaw_success_ratio * self.cfg.command_yaw_rate_max)
            )
            .float()
            .mean()
            .item()
        )
        if self._has_terrain_levels:
            extras["Curriculum/terrain_level"] = self._terrain.terrain_levels.float().mean().item()
        if self.cfg.command_curriculum:
            extras["Curriculum/command_lin_vel_max"] = self.command_lin_vel_max
        if self.cfg.dr.enabled:
            extras["DR/friction_mean"] = self.dr_friction.mean().item()
            extras["DR/delta_mass_mean"] = self.dr_delta_mass.mean().item()
            extras["DR/motor_strength_mean"] = self.dr_motor_strength.mean().item()
        self._error_xy_sum[env_ids] = 0.0
        self._error_yaw_sum[env_ids] = 0.0
        self._v_along_cmd_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        self._root_z_sum[env_ids] = 0.0
        self._speed_xy_sum[env_ids] = 0.0
        self._swing_sum[env_ids] = 0.0
        self._torque_sat_sum[env_ids] = 0.0
        self.extras["log"].update(extras)

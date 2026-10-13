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
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .hexapod_env_cfg import HexapodFlatEnvCfg, HexapodRoughEnvCfg


class HexapodEnv(DirectRLEnv):
    cfg: HexapodFlatEnvCfg | HexapodRoughEnvCfg

    def __init__(self, cfg: HexapodFlatEnvCfg | HexapodRoughEnvCfg, render_mode: str | None = None, **kwargs):
        # gait clock adds [sin, cos] to the observation (done here because Hydra overrides land after cfg init)
        base_obs = 235 if isinstance(cfg, HexapodRoughEnvCfg) else 66
        cfg.observation_space = base_obs + (2 if cfg.gait_reward_scale != 0.0 else 0)
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Per-episode tracking accumulators for the velocity-tracking success metric.
        # Errors are summed over actual episode steps and the mean is finalized at reset,
        # so logged values reflect the just-ended episode rather than the post-reset instant.
        self._error_xy_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_yaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)
        self._root_z_sum = torch.zeros(self.num_envs, device=self.device)  # logged as Metrics/root_height
        # Overnight 2026-10-12: self-reported gait metrics (episode means, flushed at reset).
        self._speed_xy_sum = torch.zeros(self.num_envs, device=self.device)  # Metrics/speed_xy: |v_xy| body frame
        self._swing_sum = torch.zeros(self.num_envs, device=self.device)  # Metrics/swing_frac: >=1 foot airborne
        self._torque_sat_sum = torch.zeros(self.num_envs, device=self.device)  # Metrics/torque_sat: joints at cap
        self._torque_cap = float(self.cfg.robot.actuators["legs"].effort_limit_sim)

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
        # tripod gait clock (exp4+)
        self._gait_on = self.cfg.gait_reward_scale != 0.0
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        _, feet_names = self._contact_sensor.find_sensors(".*_tibia")
        tripod_a = ("lf_tibia", "rm_tibia", "lr_tibia")
        self._gait_offsets = torch.tensor(
            [0.0 if n in tripod_a else math.pi for n in feet_names], device=self.device
        ).unsqueeze(0)
        self._episode_sums["gait_clock"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # toe positions: tibia links in the same name order as the contact-sensor feet
        body_ids, body_names = self._robot.find_bodies(".*_tibia")
        self._feet_body_ids = [body_ids[body_names.index(n)] for n in feet_names]
        self._toe_offset = torch.tensor([0.1225, 0.0, 0.0], device=self.device)  # club stance.json toe_local_points_m
        self._toe_z_max_sum = torch.zeros(self.num_envs, device=self.device)  # Metrics/toe_z_max
        self._prev_toe_b = torch.zeros(self.num_envs, 6, 3, device=self.device)
        self._episode_sums["stance_foot_vel"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # reference tripod (exp8+): joint index maps and per-leg sweep sign (club: left legs opposite to right legs)
        legs = ("lf", "lm", "lr", "rf", "rm", "rr")
        jn = self._robot.joint_names
        self._ref_coxa_ids = [jn.index(f"{l}_coxa_yaw") for l in legs]
        self._ref_femur_ids = [jn.index(f"{l}_femur_pitch") for l in legs]
        self._ref_tibia_ids = [jn.index(f"{l}_tibia_pitch") for l in legs]
        self._ref_side = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device).unsqueeze(0)
        self._ref_leg_offsets = torch.tensor([0.0, math.pi, 0.0, math.pi, 0.0, math.pi], device=self.device).unsqueeze(0)
        self._episode_sums["ref_gait"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, HexapodRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        if self._gait_on:
            self._gait_phase = torch.remainder(self._gait_phase + 2 * math.pi * self.step_dt / self.cfg.gait_period_s, 2 * math.pi)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch
        if self.cfg.ref_gait_residual and self._gait_on:
            self._processed_actions = self._processed_actions + self._ref_joint_delta()

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _toe_w(self) -> torch.Tensor:
        """Toe world positions for the six feet, shape (num_envs, 6, 3)."""
        pos = self._robot.data.body_pos_w.torch[:, self._feet_body_ids]
        quat = self._robot.data.body_quat_w.torch[:, self._feet_body_ids]
        return pos + quat_apply(quat, self._toe_offset.expand(pos.shape))

    def _toe_z(self) -> torch.Tensor:
        """Toe height above the ground plane for the six feet, shape (num_envs, 6)."""
        return self._toe_w()[..., 2]

    def _toe_b(self) -> torch.Tensor:
        """Toe positions in the body frame, shape (num_envs, 6, 3)."""
        rel = self._toe_w() - self._robot.data.root_pos_w.torch.unsqueeze(1)
        quat = self._robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, 6, -1)
        return quat_apply_inverse(quat, rel)

    def _ref_joint_delta(self) -> torch.Tensor:
        """Scripted tripod joint deltas [rad] for the current clock phase and command, shape (num_envs, num_joints).

        Same waveform as diag_gait_openloop.py: in the stance half the coxa sweeps +A -> -A, in the swing half it
        returns -A -> +A while femur/tibia lift with a half-sine. Sweep sign +1 moves the body along -y (Phase 0).
        """
        ph = torch.remainder((self._gait_phase.unsqueeze(1) + self._ref_leg_offsets) / (2 * math.pi), 1.0)  # (N,6)
        stance = ph < 0.5
        s_st = ph / 0.5
        s_sw = (ph - 0.5) / 0.5
        sweep = torch.where(stance, 1 - 2 * s_st, -1 + 2 * s_sw)
        lift = torch.where(stance, torch.zeros_like(ph), torch.sin(math.pi * s_sw.clamp(0, 1)))
        cmd_y = self._commands[:, 1]
        moving = (cmd_y.abs() > 0.1 * self.cfg.command_lin_vel_max).float()
        dir_sign = -torch.sign(cmd_y) * moving  # cmd -y (club forward) -> sweep sign +1
        hip = (cmd_y.abs() / 0.45).clamp(max=self.cfg.ref_hip_max)
        delta = torch.zeros(self.num_envs, len(self._robot.joint_names), device=self.device)
        delta[:, self._ref_coxa_ids] = (dir_sign * hip).unsqueeze(1) * sweep * self._ref_side
        delta[:, self._ref_femur_ids] = moving.unsqueeze(1) * self.cfg.ref_lift_femur * lift
        delta[:, self._ref_tibia_ids] = moving.unsqueeze(1) * self.cfg.ref_lift_tibia * lift
        return delta

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, HexapodRoughEnvCfg):
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
                    torch.stack([torch.sin(self._gait_phase), torch.cos(self._gait_phase)], dim=1) if self._gait_on else None,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b.torch[:, :2]), dim=1
        )
        # tolerance sigma scales with the command range, so standing still is never "close enough"
        lin_vel_sigma = self.cfg.tracking_sigma_ratio * self.cfg.command_lin_vel_max
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
            torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1 * self.cfg.command_lin_vel_max
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
        # tripod gait clock: feet airborne in their swing half, on the ground in their stance half
        toe_w = self._toe_w()
        toe_z = toe_w[..., 2]
        self._toe_z_max_sum += toe_z.max(dim=1)[0]
        # stance-foot velocity through the body frame (finite difference of body-frame toe positions)
        toe_b = self._toe_b()
        v_foot_b = (toe_b - self._prev_toe_b) / self.step_dt
        self._prev_toe_b = toe_b
        stance_foot_vel = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.stance_foot_vel_reward_scale != 0.0 and self._gait_on:
            feet_force_sv = torch.linalg.norm(self._contact_sensor.data.net_forces_w.torch[:, self._feet_ids], dim=-1)
            stance_now = (torch.sin(self._gait_phase.unsqueeze(1) + self._gait_offsets) >= 0.0) & (feet_force_sv > self.cfg.gait_stance_force_n)
            target = -self._commands[:, :2].unsqueeze(1)  # planted feet move backward through the body frame
            err_f = torch.sum(torch.square(v_foot_b[..., :2] - target), dim=-1)
            per_foot = torch.exp(-err_f / self.cfg.stance_foot_vel_sigma**2)
            n_st = stance_now.float().sum(dim=1).clamp_min(1.0)
            stance_foot_vel = (per_foot * stance_now.float()).sum(dim=1) / n_st
            stance_foot_vel = stance_foot_vel * (torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1 * self.cfg.command_lin_vel_max)
        if self._gait_on:
            feet_force = torch.linalg.norm(self._contact_sensor.data.net_forces_w.torch[:, self._feet_ids], dim=-1)
            loaded = feet_force > self.cfg.gait_stance_force_n
            if self.cfg.gait_swing_clearance_m > 0.0:
                swinging = toe_z > self.cfg.gait_toe_standing_z_m + self.cfg.gait_swing_clearance_m
            else:
                swinging = ~loaded
            # swing in the second half of each leg's cycle (sin < 0), consistent with _ref_joint_delta (fixed 02:12, after exp10 started)
            swing_expected = torch.sin(self._gait_phase.unsqueeze(1) + self._gait_offsets) < 0.0
            if self.cfg.gait_smooth and self.cfg.gait_swing_clearance_m > 0.0:
                lift = ((toe_z - self.cfg.gait_toe_standing_z_m) / self.cfg.gait_swing_clearance_m).clamp(0.0, 1.0)
                load = (feet_force / self.cfg.gait_stance_force_n).clamp(0.0, 1.0)
                gait_match = torch.where(swing_expected, lift, load).mean(dim=1) - 0.5
            else:
                gait_match = torch.where(swing_expected, swinging, loaded).float().mean(dim=1) - 0.5
            gait_match = gait_match * (torch.linalg.norm(self._commands[:, :2], dim=1) > 0.1 * self.cfg.command_lin_vel_max)
        else:
            gait_match = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.ref_gait_reward_scale != 0.0 and self._gait_on:
            q_err = self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch - self._ref_joint_delta()
            ref_gait = torch.exp(-torch.mean(torch.square(q_err), dim=1) / self.cfg.ref_gait_sigma**2)
        else:
            ref_gait = torch.zeros(self.num_envs, device=self.device)
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
            "gait_clock": gait_match * self.cfg.gait_reward_scale * self.step_dt,
            "stance_foot_vel": stance_foot_vel * self.cfg.stance_foot_vel_reward_scale * self.step_dt,
            "ref_gait": ref_gait * self.cfg.ref_gait_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Accumulate per-episode velocity tracking error sums; success is computed at reset
        # as a per-env binary check on the *episode-mean* errors against the cfg thresholds.
        error_xy = torch.linalg.norm(self._commands[:, :2] - self._robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
        error_yaw = torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b.torch[:, 2])
        self._error_xy_sum += error_xy
        self._error_yaw_sum += error_yaw
        self._step_count += 1.0
        self._root_z_sum += self._robot.data.root_pos_w.torch[:, 2]
        self._speed_xy_sum += torch.linalg.norm(self._robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
        feet_forces = torch.linalg.norm(self._contact_sensor.data.net_forces_w.torch[:, self._feet_ids], dim=-1)
        self._swing_sum += torch.any(feet_forces < 1.0, dim=1).float()
        self._torque_sat_sum += (self._robot.data.applied_torque.torch.abs() > 0.95 * self._torque_cap).float().mean(dim=1)
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

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        self._prev_toe_b[env_ids] = self._toe_b()[env_ids]
        # Sample new commands
        lin_max, yaw_max = self.cfg.command_lin_vel_max, self.cfg.command_yaw_rate_max
        if self.cfg.command_forward_only:
            mag = torch.zeros(len(env_ids), device=self.device).uniform_(self.cfg.command_lin_vel_min, lin_max)
            sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
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
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
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
        # Flush per-episode velocity tracking metrics. ``success_rate`` is per-env binary:
        # the *episode-mean* error stayed below both thresholds, mean-reduced across env_ids.
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_xy = self._error_xy_sum[env_ids] / denom
        mean_error_yaw = self._error_yaw_sum[env_ids] / denom
        extras["Metrics/error_vel_xy"] = mean_error_xy.mean().item()
        extras["Metrics/error_vel_yaw"] = mean_error_yaw.mean().item()
        extras["Metrics/root_height"] = (self._root_z_sum[env_ids] / denom).mean().item()
        extras["Metrics/speed_xy"] = (self._speed_xy_sum[env_ids] / denom).mean().item()
        extras["Metrics/swing_frac"] = (self._swing_sum[env_ids] / denom).mean().item()
        extras["Metrics/torque_sat"] = (self._torque_sat_sum[env_ids] / denom).mean().item()
        extras["Metrics/toe_z_max"] = (self._toe_z_max_sum[env_ids] / denom).mean().item()
        extras["Metrics/success_rate"] = (
            (
                (mean_error_xy < self.cfg.vel_xy_success_ratio * self.cfg.command_lin_vel_max)
                & (mean_error_yaw < self.cfg.vel_yaw_success_ratio * self.cfg.command_yaw_rate_max)
            )
            .float()
            .mean()
            .item()
        )
        self._error_xy_sum[env_ids] = 0.0
        self._error_yaw_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        self._root_z_sum[env_ids] = 0.0
        self._speed_xy_sum[env_ids] = 0.0
        self._swing_sum[env_ids] = 0.0
        self._torque_sat_sum[env_ids] = 0.0
        self._toe_z_max_sum[env_ids] = 0.0
        self.extras["log"].update(extras)

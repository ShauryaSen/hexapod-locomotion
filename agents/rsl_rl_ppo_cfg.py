# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class HexapodFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "hexapod_flat_direct"
    # the env also returns a "privileged" group (true DR values); plain PPO uses only "policy" for actor and critic
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HexapodRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hexapod_rough_direct"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # Same 128x3 layout as the flat runner (was ANYmal's 512/256/128) so stage 1 can warm-start from a flat sweep
    # checkpoint with --resume (decisions.md P7: staged training). Default choice, 2026-10-21.
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RMATeacherModelCfg(RslRlMLPModelCfg):
    """hexapod.rma.RMATeacherModel: MLP on [policy obs, z], z = mu(privileged e) (decisions.md P6/P7)."""

    class_name: str = "hexapod.rma:RMATeacherModel"
    priv_group: str = "privileged"
    z_dim: int = 8
    """CITED: RMA (arXiv 2107.04034) Sec. III-A, z in R^8 (verified 2026-10-23; A1 quadruped, e is 17-dim there)."""
    encoder_hidden_dims: list[int] = [256, 128]
    """CITED: RMA Sec. IV-B, mu is a 3-layer MLP with hidden sizes (256, 128). Our e is only 3-dim, so this is
    larger than needed; kept to follow the reference (verified 2026-10-23)."""
    priv_center: list[float] = [0.0, 0.875, 1.0]
    priv_half_range: list[float] = [0.75, 0.375, 0.2]
    """ASSUMED: fixed scaling of e to about [-1, 1] over the D1-D3 training ranges (see hexapod/rma.py)."""


@configclass
class HexapodRoughRMATeacherPPORunnerCfg(HexapodRoughPPORunnerCfg):
    """Stage 2 (P7): same PPO recipe as stage 1, actor and critic see z = mu(e). Warm start from a padded stage-1
    checkpoint (tools/rma_init.py), so --resume continues the iteration counter."""

    max_iterations = 2000
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged"]}
    actor = RMATeacherModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RMATeacherModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )

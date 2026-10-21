# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the deployment harness. Plain dataclasses (no Isaac Lab import) so every tool can build and
save them without Kit. Every field that is not fixed by decisions.md carries its label in the docstring:
MEASURED (from a run), CITED (paper / doc), ASSUMED (a default, unverified)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


def to_dict(obj) -> dict:
    return dataclasses.asdict(obj)


def from_dict(cls, d: dict):
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if dataclasses.is_dataclass(f.type) if isinstance(f.type, type) else False:
            v = from_dict(f.type, v)
        kw[f.name] = v
    return cls(**kw)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(to_dict(obj) if dataclasses.is_dataclass(obj) else obj, f, indent=1)


@dataclass
class DeployCfg:
    """One batch of N deployments in one Isaac Lab process."""

    task: str = "Hexapod-Rough-Direct-v0"
    cmd_speed: float = 0.28
    """Fixed forward command [m/s] (decisions.md F3, decided)."""
    gait_cycle_s: float = 0.4
    """Stream resolution = one gait cycle (F5, decided; the policy's clock period is 0.4 s, MEASURED in the sweep)."""
    levels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    """Terrain levels the deployments are pinned to (P4: deployments run on L0-L5, decided)."""
    reposition_s: float = 15.0
    """Teleport every robot back to its terrain cell every this many seconds, not a fall (tools/long_run.py value:
    robots walk off the 10 m terrain in ~40 s at 0.28 m/s; MEASURED 2026-10-21). 0 = never."""
    fall_recovery_s: float = 5.0
    """Fixed scripted stand-up time (R4): after a fall the robot is reset at its cell and held in the default pose
    for this long: no progress, no samples for the updater, no residual. ASSUMED (R4 says: set in the harness)."""
    cap_progress_at_cmd: bool = True
    """Progress per step = min(v_along_cmd, cmd) x dt (R4: progress capped at the commanded speed)."""
    explore_only_when_adapting: bool = True
    """Act with the Gaussian sample only while a deployment is adapting (PPO needs on-policy samples); act with the
    mean while holding. ASSUMED: exploration noise is a real cost of on-robot adaptation, so it is charged to the
    arm that adapts, not to every arm."""
    env_seed: int = 0
    """Seed for the env and the policy sampling (per-deployment generators are derived from it)."""
    env_override: tuple[str, ...] = ()
    """key=value env cfg overrides, applied after <run>/params/env.yaml."""


@dataclass
class UpdaterCfg:
    """KL-anchored PPO fine-tune (F7). LR and KL weight are nuisance knobs to be tuned once in the kill test."""

    steps_per_batch: int = 500
    """Robot steps (0.02 s each) accumulated per PPO update of one deployment = 10 s of robot time. ASSUMED: rsl_rl
    trains on 24 steps x 512 envs; one robot needs a longer window for a usable advantage estimate."""
    num_learning_epochs: int = 5
    """rsl_rl training value (agents/rsl_rl_ppo_cfg.py)."""
    num_mini_batches: int = 4
    """rsl_rl training value; mini-batches are contiguous time chunks of one deployment's buffer."""
    learning_rate: float = 1.0e-4
    """Task brief default (ASSUMED; rsl_rl training used 1e-3 adaptive)."""
    lr_schedule: str = "fixed"
    """'fixed' or 'adaptive' (rsl_rl rule on desired_kl, per deployment). ASSUMED fixed for a fine-tune."""
    desired_kl: float = 0.01
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    entropy_coef: float = 0.005
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    """Per-deployment gradient-norm clip (rsl_rl value)."""
    kl_anchor_weight: float = 1.0
    """Weight of KL(pi_theta || pi_0) on the batch states, added to the PPO loss. ASSUMED 1.0 (nuisance knob, F7)."""
    budget_min: float = 4.0
    """Robot-minutes one adaptation episode may spend adapting = 2 x T50 (F7). Placeholder until T50 is MEASURED
    (tools/deploy_t50.py); 4.0 is ASSUMED for smoke tests."""
    update_critic: bool = True
    train_std: bool = True
    """Whether the Gaussian std is fine-tuned too (rsl_rl trains it). ASSUMED yes."""


@dataclass
class EnsembleCfg:
    """Dynamics ensemble (F5): 5 MLPs (obs, action) -> next proprioceptive obs delta."""

    n_members: int = 5
    hidden_dims: tuple[int, ...] = (256, 256)
    """ASSUMED."""
    lr: float = 1.0e-3
    batch_size: int = 4096
    epochs: int = 30
    weight_decay: float = 0.0
    train_fraction: float = 0.9
    """The rest is held out: nominal residual mean / sigma for calibration come from it (not from training data)."""
    target: str = "proprio_delta"
    """Predict next (lin vel, ang vel, projected gravity, joint pos, joint vel) minus current = the 45 proprioceptive
    dims; commands, previous actions and the clock are inputs only. ASSUMED."""
    residual: str = "nmse"
    """Per-step residual = mean over target dims of ((ensemble-mean prediction - target) / sigma_target)^2; per cycle =
    mean over the cycle's steps. ASSUMED. 'disagreement' (variance across members) is logged alongside as the
    ablation signal."""
    residual_transform: str = "log"
    """Transform applied to the per-cycle residual before standardization: 'log' or 'none'. MEASURED 2026-10-21 on
    32 nominal robot-minutes (seed 1, DR range, L0-L5): the raw per-cycle nMSE has skew 7.8 and max z 19.7 (p99
    4.1); log(nMSE) has skew 0.7 and max z 4.8 (p99 3.0). The raw residual stays in the streams; only z changes."""


@dataclass
class DetectorCfg:
    """Detectors on the standardized per-cycle residual z = (r - mu_nominal) / sigma_nominal."""

    three_sigma_k: float = 3.0
    """Threshold in nominal sigmas (Domberg-style 3-sigma rule, F6)."""
    three_sigma_consecutive: int = 3
    """Cycles in a row above the threshold before an alarm. ASSUMED 3 (1.2 s) against single-cycle spikes; 1 = the
    literal rule."""
    cusum_k: float = 0.5
    """CUSUM reference value (drift) in sigma units: S = max(0, S + z - k). ASSUMED 0.5 (the textbook choice for a
    1-sigma shift)."""
    cusum_h: float = 5.0
    """CUSUM alarm threshold; replaced by tools/deploy_calibrate_cusum.py (target false alarms/hour on held-out
    nominal runs). 5.0 is a textbook placeholder (ASSUMED)."""
    cusum_target_fa_per_hour: float = 1.0
    """Calibration target. ASSUMED 1/h (same order as the P5 fall bar)."""
    converge_band: float = 2.0
    """Convergence stop: |z| below this many nominal sigmas ..."""
    converge_cycles: int = 25
    """... for this many consecutive cycles (25 x 0.4 s = 10 s) ends the adaptation episode. ASSUMED."""
    baseline_cycles: int = 150
    """R5: the nominal residual level is (re-)measured per deployment over this many cycles (150 = 60 s, ASSUMED)
    at the start of the deployment and after every adaptation episode; z = (log r - mu_deployment) / sigma_global;
    no alarms while measuring. 0 = global calibration only."""


@dataclass
class ScheduleCfg:
    """Change schedules (E1)."""

    t50_min: float = 2.0
    """T50 in robot-minutes. Placeholder (ASSUMED) until MEASURED by tools/deploy_t50.py; the generator scales
    everything from it."""
    mean_gap_factor: float = 8.0
    """Mean Poisson gap = this x T50 (E1, decided)."""
    n_changes_target: int = 6
    """~6 changes per deployment (E1): deployment length = n_changes_target x mean gap; schedules whose Poisson draw
    has fewer than n_changes_min or more than n_changes_max changes are redrawn."""
    n_changes_min: int = 4
    n_changes_max: int = 8
    min_nominal_fraction: float = 0.4
    """>= 40% of the deployment with every factor off (E1). Rejection sampled."""
    require_recurrence: bool = True
    """At least one factor switches on twice (E1: 'so regimes recur'). Rejection sampled."""
    warmup_s: float = 60.0
    """Nominal time before the first change can arrive (detector baseline, T0 of the progress rate). ASSUMED."""
    payload_kg: float = 2.5
    """D3 test value."""
    motor_strength: float = 0.5
    """D2 test value (0.5 vs 0.6 to be chosen in the kill test)."""
    friction: float = 0.175
    """D1 test value: the middle of the 0.15-0.2 slippery patch. ASSUMED."""
    factors: tuple[str, ...] = ("payload", "motor", "friction")
    tune_seeds: tuple[int, ...] = tuple(range(0, 10))
    """R2: schedules used for choosing each arm's knob."""
    test_seeds: tuple[int, ...] = tuple(range(10, 20))
    """R2: held-out schedules; hypotheses are tested on these with paired differences."""


@dataclass
class ArmCfg:
    """Which arm and its parameters. Only the fields the arm uses matter."""

    name: str = "never"
    """never | always | scheduled | periodic_reset | pause_hold | three_sigma | cusum | triggered | oracle"""
    period_min: float = 10.0
    """scheduled(T) / periodic_reset(T): period in robot-minutes (the arm's knob, R6)."""
    cusum_h: float | None = None
    """cusum: alarm threshold (the arm's knob, R6: set from the target false-alarm rate); None = DetectorCfg."""
    three_sigma_consecutive: int | None = None
    """three_sigma: cycles above threshold before an alarm (knob); None = DetectorCfg."""
    hold_min: float = 10.0
    """pause_hold: hold time after each budget of adaptation."""
    budget_min: float | None = None
    """Override of UpdaterCfg.budget_min for this arm (None = the updater's)."""


@dataclass
class MetricsCfg:
    detect_window_min: float | None = None
    """A change counts as detected if an adaptation starts within this window after it (None = the budget). ASSUMED."""
    recurrence_window_min: float | None = None
    """Window after an onset over which the recurrence gain is measured (None = the budget). ASSUMED."""
    bootstrap_n: int = 1000
    ci: float = 0.95

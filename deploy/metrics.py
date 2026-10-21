# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metrics (decisions.md E2) from one deployment's streams and its schedule.

Per deployment:
  progress_m          metres along the command over the deployment (falls cost their recovery time: 0 progress); the ranking metric
  progress_per_h      the same per robot-hour
  falls, falls_per_h
  track_err           mean tracking error over walking cycles [m/s]
  adapt_min           robot-minutes spent adapting
  false_alarms, false_alarms_per_h   adaptation starts (alarm cycles) with no true change in the preceding window
  detection_delay_s   per true change: time to the first alarm within the window (NaN = missed); mean over changes
  missed              changes with no alarm within the window
  reward_sum          summed task reward over the deployment (reported alongside progress; E2 update 2026-10-21:
                      no regret vs the oracle, the oracle is one arm in the table)
  quiet_progress_m / quiet_min        progress on quiet stretches: more than one window after the last change
  post_change_progress_m / post_change_min   progress within one window after any change
  recurrence_progress_m / recurrence_min      progress within one window after a 2nd+ onset of a factor (subset of
                      post-change)
  recurrence_gain_m   progress in the window after the 2nd onset of a factor minus the same after its 1st onset
                      (negative = forgetting), mean over recurring factors; NaN if nothing recurs
Across arms (summarizer): quiet-stretch progress vs the never arm, matched per (checkpoint, schedule, env); progress
and reward rankings with a flag when they disagree; bootstrap CIs over (seed x schedule) deployments.
"""

from __future__ import annotations

import math

import numpy as np

from .cfg import MetricsCfg
from .schedule import Schedule


def _nan(x):
    return float("nan") if x is None else x


def deployment_metrics(arrays: dict[str, np.ndarray], i: int, schedule: Schedule | None, cfg: MetricsCfg, budget_min: float) -> dict:
    t = arrays["t_s"][:, i]
    if len(t) == 0:
        return {}
    cycle_s = float(t[1] - t[0]) if len(t) > 1 else float(t[0])
    hours = t[-1] / 3600.0
    prog = arrays["progress_m"][:, i]
    falls = arrays["falls"][:, i]
    adapting = arrays["adapting"][:, i] > 0.5
    alarm = arrays["alarm"][:, i] > 0.5
    track = arrays["track_err"][:, i]
    reward = arrays["reward"][:, i] if "reward" in arrays else np.full_like(prog, np.nan)
    out = {
        "minutes": t[-1] / 60.0,
        "progress_m": float(np.nansum(prog)),
        "progress_per_h": float(np.nansum(prog) / max(hours, 1e-9)),
        "falls": float(np.nansum(falls)),
        "falls_per_h": float(np.nansum(falls) / max(hours, 1e-9)),
        "reward_sum": float(np.nansum(reward)),
        "track_err": float(np.nanmean(track)) if np.any(~np.isnan(track)) else float("nan"),
        "adapt_min": float(adapting.sum() * cycle_s / 60.0),
        "alarms": int(alarm.sum()),
    }
    window_s = (cfg.detect_window_min if cfg.detect_window_min is not None else budget_min) * 60.0
    rec_window_s = (cfg.recurrence_window_min if cfg.recurrence_window_min is not None else budget_min) * 60.0
    alarm_times = t[alarm]
    if schedule is not None:
        changes = np.array(schedule.change_times())
        # false alarms: alarms with no change in (t - window, t]
        fa = 0
        for ta in alarm_times:
            if not np.any((changes <= ta) & (changes > ta - window_s)):
                fa += 1
        delays, missed = [], 0
        for tc in changes:
            after = alarm_times[(alarm_times >= tc) & (alarm_times <= tc + window_s)]
            if len(after):
                delays.append(after[0] - tc)
            else:
                missed += 1
        # segments: post-change = within one window after any change; recurrence = within one window after a
        # 2nd+ onset of a factor; quiet = the rest (incl. the warm-up)
        post = np.zeros(len(t), dtype=bool)
        for tc in changes:
            post |= (t > tc) & (t <= tc + rec_window_s)
        rec = np.zeros(len(t), dtype=bool)
        gains = []
        for f in ("payload", "motor", "friction"):
            on = schedule.onsets(f)
            for tc in on[1:]:
                rec |= (t > tc) & (t <= tc + rec_window_s)
            if len(on) >= 2:
                w1 = prog[(t > on[0]) & (t <= on[0] + rec_window_s)].sum()
                w2 = prog[(t > on[1]) & (t <= on[1] + rec_window_s)].sum()
                gains.append(w2 - w1)
        quiet = ~post
        nominal = (arrays["payload_kg"][:, i] == 0) & (arrays["motor_strength"][:, i] == 1) & (np.abs(arrays["friction"][:, i] - arrays["friction"][0, i]) < 1e-6)
        out.update({
            "quiet_progress_m": float(prog[quiet].sum()), "quiet_min": float(quiet.sum() * cycle_s / 60.0),
            "quiet_progress_per_h": float(prog[quiet].sum() / (quiet.sum() * cycle_s / 3600.0)) if quiet.any() else float("nan"),
            "post_change_progress_m": float(prog[post].sum()), "post_change_min": float(post.sum() * cycle_s / 60.0),
            "recurrence_progress_m": float(prog[rec].sum()), "recurrence_min": float(rec.sum() * cycle_s / 60.0),
            "nominal_fraction": float(nominal.mean()),
            "false_alarms": fa,
            "false_alarms_per_h": fa / max(hours, 1e-9),
            "detection_delay_s": float(np.mean(delays)) if delays else float("nan"),
            "missed": missed,
            "n_changes": len(changes),
            "recurrence_gain_m": float(np.mean(gains)) if gains else float("nan"),
        })
    else:
        out.update({"false_alarms": int(alarm.sum()), "false_alarms_per_h": alarm.sum() / max(hours, 1e-9), "detection_delay_s": float("nan"), "missed": 0, "n_changes": 0, "quiet_progress_m": out["progress_m"], "quiet_min": out["minutes"], "quiet_progress_per_h": out["progress_per_h"], "post_change_progress_m": 0.0, "post_change_min": 0.0, "recurrence_progress_m": 0.0, "recurrence_min": 0.0, "nominal_fraction": 1.0, "recurrence_gain_m": float("nan")})
    return out


def bootstrap_ci(x, n_boot: int = 1000, ci: float = 0.95, rng=None) -> tuple[float, float]:
    x = np.asarray([v for v in np.asarray(x, dtype=float) if not math.isnan(v)])
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = rng or np.random.default_rng(0)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def fmt_ci(x, n_boot=1000, ci=0.95) -> str:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "–"
    lo, hi = bootstrap_ci(x, n_boot, ci)
    return f"{x.mean():.2f} [{lo:.2f}, {hi:.2f}]"

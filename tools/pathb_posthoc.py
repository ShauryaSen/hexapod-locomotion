# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path B POST HOC, EXPLORATORY analyses (supervisor 11-12, decisions.md; nothing here changes the pre-registered
result, which is deploy_score_detectors' score.json). CPU and numpy only. Reuses the scorer's own loading,
standardisation, detectors and scoring rules so every definition matches the locked one.

  (a) secondary table: type means without friction; CUSUM target 0.5 (h calibrated on tune as in the scorer) scored
      on test, chosen AFTER seeing test
  (b) z traces around onset and offset per change type (test split)
  (c) false alarms by position: time since the last switch back to nominal, tune vs test
  (d) report figures (PNG + PDF), including the adaptation-failure summary

  python tools/pathb_posthoc.py --jobs <job_00/run> ... --score <score.json> --out <prefix> --fig_dir <dir> \
      [--replay_verdict verdict.json] [--fleet_jobs <job_00> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_score_detectors import FACTORS, alarms_cusum, alarms_three_sigma, fa_rate, load_chain, score, segments, standardize, world_state  # noqa: E402

CYCLE_S = 0.4
LABEL = {"payload": "payload 6 kg", "motor": "motor 0.5", "friction": "friction 0.1"}


# ---- schedule-cluster bootstrap on per-seed sums ----------------------------------------------------------------


def seed_weights(n_seeds: int, boot: int, rng) -> np.ndarray:
    """(boot, n_seeds) multiplicity of each schedule in each resample."""
    pick = rng.integers(0, n_seeds, size=(boot, n_seeds))
    return np.stack([np.bincount(p, minlength=n_seeds) for p in pick]).astype(float)


def ratio_ci(num: np.ndarray, den: np.ndarray, W: np.ndarray):
    """num, den: (n_seeds, ...) per-schedule sums. Point ratio and 95% CI of sum(num)/sum(den) over resamples."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = np.where(den.sum(0) > 0, num.sum(0) / np.maximum(den.sum(0), 1e-12), np.nan)
    bn = np.tensordot(W, num, axes=(1, 0))
    bd = np.tensordot(W, den, axes=(1, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = bn / bd
    lo, hi = np.nanquantile(b, 0.025, axis=0), np.nanquantile(b, 0.975, axis=0)
    return pt, lo, hi, b


def onset_metrics(sc: dict, robots: np.ndarray, seed_of: np.ndarray, seeds: list, W: np.ndarray) -> dict:
    """Per type miss rate and censored delay; type means with and without friction; FA/h. All with CIs."""
    rset = set(robots.tolist())
    si = {s: k for k, s in enumerate(seeds)}
    S = len(seeds)
    n, miss, cens, det_n, det_sum = (np.zeros((S, 3)) for _ in range(5))
    for o in sc["onsets"]:
        if o[0] not in rset:
            continue
        k, f = si[seed_of[o[0]]], o[1]
        n[k, f] += 1
        cens[k, f] += o[3]
        if o[2] is None:
            miss[k, f] += 1
        else:
            det_n[k, f] += 1
            det_sum[k, f] += o[2]
    fa = np.array([sc["fa"][robots][seed_of[robots] == s].sum() for s in seeds])
    hrs = np.array([sc["hours"][robots][seed_of[robots] == s].sum() for s in seeds])
    out = {}
    p, lo, hi, _ = ratio_ci(fa, hrs, W)
    out["fa_per_hour"] = [float(p), float(lo), float(hi)]
    pm, lm, hm, _ = ratio_ci(miss, n, W)
    pc, lc, hc, bc = ratio_ci(cens, n, W)
    pdl, ldl, hdl, _ = ratio_ci(det_sum, det_n, W)
    for f, name in enumerate(FACTORS):
        out[name] = {"n": int(n[:, f].sum()), "miss_rate": [float(pm[f]), float(lm[f]), float(hm[f])], "censored_delay_s": [float(pc[f]), float(lc[f]), float(hc[f])],
                     "delay_if_detected_s": [float(pdl[f]), float(ldl[f]), float(hdl[f])] if det_n[:, f].sum() else None}
    for key, idx in (("type_mean_s", [0, 1, 2]), ("type_mean_no_friction_s", [0, 1])):
        b = bc[:, idx].mean(1)
        out[key] = [float(pc[idx].mean()), float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))]
    out["_boot_type_mean_no_friction"] = bc[:, [0, 1]].mean(1)
    out["_boot_type_mean"] = bc.mean(1)
    return out


# ---- (c) false alarms by position --------------------------------------------------------------------------------

POS_BINS = [("before the first onset", None, None), ("10-20 s after an offset", 10.0, 20.0), ("20-40 s after an offset", 20.0, 40.0), (">= 40 s after an offset", 40.0, np.inf)]


def fa_by_position(alarm, run_start, st, z, settle_s=10.0):
    """Per robot and position bin: false alarms and scored nominal hours, with exactly score()'s exclusions
    (baseline = NaN z, settle window, carry-over alarms). Returns (fa, hours) of shape (n_robots, n_bins)."""
    T, n = st.shape
    settle_c = int(round(settle_s / CYCLE_S))
    valid = ~np.isnan(z)
    fa = np.zeros((n, len(POS_BINS)))
    hrs = np.zeros((n, len(POS_BINS)))
    for i in range(n):
        excl = np.zeros(T, dtype=bool)
        last_end = np.full(T, -1)
        for f, a, e in segments(st[:, i]):
            last_end[e:] = e
            excl[e : e + settle_c] = True
        nominal = (st[:, i] < 0) & valid[:, i]
        co = alarm[:, i] & nominal & (run_start[:, i] >= 0) & (run_start[:, i] < last_end)
        scored = nominal & ~co & ~excl
        since = np.where(last_end >= 0, (np.arange(T) - last_end) * CYCLE_S, np.nan)  # s since the first nominal cycle
        for b, (_, lo, hi) in enumerate(POS_BINS):
            m = scored & (np.isnan(since) if lo is None else ((since >= lo) & (since < hi)))
            fa[i, b] = alarm[m, i].sum()
            hrs[i, b] = m.sum() * CYCLE_S / 3600.0
    return fa, hrs


# ---- (b) z traces -------------------------------------------------------------------------------------------------


def z_traces(z, st, robots, seed_of, seeds, W, pre_c=25, post_on_c=137, post_off_c=75):
    """Per type: z aligned at onset (lags -pre_c..post_on_c, masked after the segment ends) and at offset (lags
    -pre_c..post_off_c, masked before the segment starts). Mean over episodes with a schedule-cluster CI, and the
    episode quartiles."""
    T = st.shape[0]
    si = {s: k for k, s in enumerate(seeds)}
    out = {}
    for align, lags in (("onset", np.arange(-pre_c, post_on_c + 1)), ("offset", np.arange(-pre_c, post_off_c + 1))):
        for f, name in enumerate(FACTORS):
            rows, sk = [], []
            for i in robots:
                for ff, a, e in segments(st[:, i]):
                    if ff != f:
                        continue
                    ref = a if align == "onset" else e
                    idx = ref + lags
                    v = np.full(len(lags), np.nan)
                    ok = (idx >= 0) & (idx < T)
                    v[ok] = z[idx[ok], i]
                    if align == "onset":
                        v[idx >= e] = np.nan
                    else:
                        v[idx < a] = np.nan
                    rows.append(v)
                    sk.append(si[seed_of[i]])
            R = np.array(rows)
            sk = np.array(sk)
            num = np.stack([np.nansum(R[sk == k], 0) for k in range(len(seeds))])
            den = np.stack([(~np.isnan(R[sk == k])).sum(0) for k in range(len(seeds))]).astype(float)
            p, lo, hi, _ = ratio_ci(num, den, W)
            with np.errstate(all="ignore"):
                q25, q75 = np.nanquantile(R, 0.25, axis=0), np.nanquantile(R, 0.75, axis=0)
            out[(align, name)] = {"t_s": lags * CYCLE_S, "mean": p, "lo": lo, "hi": hi, "q25": q25, "q75": q75, "n_episodes": len(R), "frac_valid": (~np.isnan(R)).mean(0)}
    return out


def on_segment_z(z, st, robots, seed_of, seeds, W):
    """Per type, mean z over all valid on-segment cycles and the fraction of them above 3 and above k = 0.5 (test)."""
    si = {s: k for k, s in enumerate(seeds)}
    res = {}
    for f, name in enumerate(FACTORS):
        num, den, a3, ak = (np.zeros(len(seeds)) for _ in range(4))
        for i in robots:
            m = (st[:, i] == f) & ~np.isnan(z[:, i])
            k = si[seed_of[i]]
            num[k] += z[m, i].sum()
            den[k] += m.sum()
            a3[k] += (z[m, i] > 3).sum()
            ak[k] += (z[m, i] > 0.5).sum()
        res[name] = {"mean_z": [float(x) for x in ratio_ci(num, den, W)[:3]], "frac_above_3": [float(x) for x in ratio_ci(a3, den, W)[:3]], "frac_above_k": [float(x) for x in ratio_ci(ak, den, W)[:3]]}
    nom_num, nom_den = np.zeros(len(seeds)), np.zeros(len(seeds))
    for i in robots:
        m = (st[:, i] < 0) & ~np.isnan(z[:, i])
        nom_num[si[seed_of[i]]] += z[m, i].sum()
        nom_den[si[seed_of[i]]] += m.sum()
    res["nominal"] = {"mean_z": [float(x) for x in ratio_ci(nom_num, nom_den, W)[:3]]}
    return res


# ---- adaptation-failure evidence (existing logs) ------------------------------------------------------------------


def replay_summary(path, rng, boot):
    v = json.load(open(path))
    fr = v["frozen"]
    rf = []  # replay - frozen per replicate in replicate order (even block = replicates 0, 2, ...)
    for r in range(12):
        rf.append(v["blocks"]["even" if r % 2 == 0 else "odd"]["per_deployment"][r // 2])
    rf = np.array(rf)
    ro = np.array(v["paired"]["per_replicate"])  # replay - onpolicy
    of = rf - ro  # onpolicy - frozen
    se_fr = float(np.sqrt(fr * (1 - fr) / 216))

    def ci(x):
        b = x[rng.integers(0, len(x), size=(boot, len(x)))].mean(1)
        return [float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]

    return {"frozen": fr, "frozen_se": se_fr, "bar": v["bar"], "replay_minus_frozen": ci(rf), "onpolicy_minus_frozen": ci(of), "replay_minus_onpolicy": v["paired"]["mean"], "replay_minus_onpolicy_ci": v["paired"]["ci"],
            "blocks": {b: v["blocks"][b]["replay_minus_frozen"] for b in ("even", "odd")}, "verdict": v["verdict"]}


def fleet_summary(jobs, rng, boot, t_on=30.0, group="onpolicy", control="control"):
    prog, ts, offset, meta0 = [], [], 0.0, None
    for j in jobs:
        zz = np.load(os.path.join(j, "streams.npz"))
        m = json.load(open(os.path.join(j, "meta.json")))
        meta0 = meta0 or m
        prog.append(zz["progress_m"])
        ts.append(zz["t_s"][:, 0] + offset)
        offset += float(zz["t_s"][-1, 0])
    prog, t = np.concatenate(prog), np.concatenate(ts)
    dt = t[1] - t[0]
    names = [g["name"] if isinstance(g, dict) else None for g in meta0["groups"]] if isinstance(meta0["groups"], list) else None
    if names is None or None in names:
        names = ["control", "onpolicy"]  # meta stores the groups as a string; order as group_of_env (0 control, 1 onpolicy)
    gi, ci_ = names.index(group), names.index(control)
    genv, renv = np.array(meta0["group_of_env"]), np.array(meta0["replicate_of_env"])
    cells = np.array(meta0["types_per_env"]) * 10 + np.array(meta0["levels_per_env"])
    after, before = t > t_on + 1e-6, t <= t_on + 1e-6
    minutes_after = after.sum() * dt / 60.0
    out = []
    for r in sorted(set(renv.tolist())):
        fl = np.where((genv == gi) & (renv == r))[0]
        co = np.where((genv == ci_) & (renv == r))[0]
        assert (cells[fl] == cells[co]).all(), "fleet and control robots are not on the same cells in the same order"
        rate = lambda idx, sel: prog[sel][:, idx].sum(0) / (sel.sum() * dt / 60.0)  # per robot m/min
        R0 = 0.5 * (rate(fl, before).mean() + rate(co, before).mean())
        Rf = rate(co, after).mean()
        d = rate(fl, after) - rate(co, after)  # paired by cell, whole 20 min after the change
        b = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
        lo, hi = np.quantile(b, [0.025, 0.975])
        out.append({"seed": int(r), "R0": float(R0), "Rf": float(Rf), "minutes_after": float(minutes_after), "fleet_minus_control": [float(d.mean()), float(lo), float(hi)],
                    "gap_closed": [float(d.mean() / (R0 - Rf)), float(lo / (R0 - Rf)), float(hi / (R0 - Rf))], "n_cells": int(len(d))})
    return out


# ---- figures ------------------------------------------------------------------------------------------------------

BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "savefig.dpi": 200})
    return plt


def save(fig, fig_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight")


def fig_table(plt, primary, fig_dir):
    t = primary["test"]

    def cell(m, pct=False, p=1):
        if m is None:
            return "none detected"
        return f"{100 * m[0]:.0f}% [{100 * m[1]:.0f}, {100 * m[2]:.0f}]" if pct else f"{m[0]:.{p}f} [{m[1]:.{p}f}, {m[2]:.{p}f}]"

    def g(det, f, key):
        x = t[det][f]
        k = {"miss": ("miss_rate", "miss_rate_ci"), "cens": ("censored_mean_delay_s", "censored_mean_delay_s_ci"), "det": ("mean_delay_detected_s", "mean_delay_detected_s_ci")}[key]
        if x[k[0]] is None:
            return None
        return [x[k[0]], *(x.get(k[1]) or [x[k[0]], x[k[0]]])]

    cols = ["", "3σ, m = 3", "CUSUM, h = 8.75"]
    rows = [["false alarms / h", cell([t["three_sigma"]["fa_per_hour"], *t["three_sigma"]["fa_per_hour_ci"]], p=2), cell([t["cusum"]["fa_per_hour"], *t["cusum"]["fa_per_hour_ci"]], p=2)]]
    for f in FACTORS:
        rows.append([f"{LABEL[f]}: missed", cell(g("three_sigma", f, "miss"), pct=True), cell(g("cusum", f, "miss"), pct=True)])
        rows.append([f"{LABEL[f]}: censored delay (s)", cell(g("three_sigma", f, "cens")), cell(g("cusum", f, "cens"))])
        rows.append([f"{LABEL[f]}: delay if detected (s)", cell(g("three_sigma", f, "det")), cell(g("cusum", f, "det"))])
    rows.append(["censored delay, type mean (s)", cell([t["three_sigma"]["censored_mean_delay_s_types"], *t["three_sigma"]["censored_mean_delay_s_types_ci"]]), cell([t["cusum"]["censored_mean_delay_s_types"], *t["cusum"]["censored_mean_delay_s_types_ci"]])])
    dd = t["cusum_minus_three_sigma_censored_delay_s"]
    rows.append(["CUSUM − 3σ, type mean (s, paired)", "", f"{dd['mean']:+.1f} [{dd['ci'][0]:+.1f}, {dd['ci'][1]:+.1f}]"])
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * (len(rows) + 1) + 0.9))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colWidths=[0.42, 0.29, 0.29])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.35)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.set_linewidth(0.6)
        if r == 0:
            cl.set_text_props(weight="bold", color=INK)
            cl.set_facecolor("#f0efec")
        elif r == 1 and c == 2:
            cl.set_text_props(weight="bold")
    ax.set_title("Path B, test split (10 schedules × 18 robots), knobs frozen from tune\n95% schedule-cluster bootstrap CIs; pre-registered primary result", fontsize=9.5, color=INK, loc="left")
    save(fig, fig_dir, "pathB_detector_table")
    plt.close(fig)


def fig_delay_miss(plt, primary, fig_dir):
    t = primary["test"]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
    x = np.arange(3)
    w = 0.36
    for ax, key, cik, ylab, scale in ((axs[0], "miss_rate", "miss_rate_ci", "episodes missed (%)", 100), (axs[1], "censored_mean_delay_s", "censored_mean_delay_s_ci", "censored delay (s)", 1)):
        for j, (det, name, col) in enumerate((("three_sigma", "3σ, m = 3", BLUE), ("cusum", "CUSUM, h = 8.75", ORANGE))):
            v = np.array([t[det][f][key] for f in FACTORS]) * scale
            c = np.array([t[det][f][cik] for f in FACTORS]) * scale
            ax.bar(x + (j - 0.5) * (w + 0.02), v, w, color=col, label=name, edgecolor="white", linewidth=1)
            ax.errorbar(x + (j - 0.5) * (w + 0.02), v, yerr=[v - c[:, 0], c[:, 1] - v], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
        ax.set_xticks(x, [LABEL[f] for f in FACTORS])
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylim(0, 105)
    axs[0].legend(loc="upper left", fontsize=8)
    axs[1].set_title("missed episodes count at their full on-length (35–55 s)", fontsize=8, color=INK2, loc="left")
    fig.suptitle("Path B test split: misses and delay per change type (95% schedule-cluster CIs)", fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, fig_dir, "pathB_delay_miss")
    plt.close(fig)


def fig_z(plt, tr, fig_dir):
    fig, axs = plt.subplots(2, 3, figsize=(9.0, 4.8), sharey=True)
    for c, f in enumerate(FACTORS):
        for r, align in enumerate(("onset", "offset")):
            ax = axs[r, c]
            d = tr[(align, f)]
            ax.fill_between(d["t_s"], d["q25"], d["q75"], color=BLUE, alpha=0.13, linewidth=0, label="episode quartiles")
            ax.fill_between(d["t_s"], d["lo"], d["hi"], color=BLUE, alpha=0.35, linewidth=0, label="95% CI of the mean")
            ax.plot(d["t_s"], d["mean"], color=BLUE, lw=1.6, label="mean z")
            ax.axvline(0, color=INK2, lw=0.8)
            ax.axhline(3, color=INK2, lw=0.8, ls="--")
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
            ax.axhline(0, color=GRID, lw=0.8)
            if r == 0:
                ax.set_title(f"{LABEL[f]} (n = {d['n_episodes']})", fontsize=9, color=INK)
            ax.set_xlabel(f"s from {align}")
            if c == 0:
                ax.set_ylabel("z")
    axs[0, 2].text(54, 3.08, "3σ line", fontsize=7, color=INK2, ha="right", va="bottom")
    axs[0, 2].text(54, 0.58, "CUSUM k", fontsize=7, color=INK2, ha="right", va="bottom")
    axs[0, 0].legend(loc="upper left", fontsize=7)
    axs[0, 0].set_ylim(-2, 6)
    fig.suptitle("Residual z around a change, test split (POST HOC, exploratory). Onset row: on-segment only after 0; offset row: nominal only after 0", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, fig_dir, "pathB_z_traces")
    plt.close(fig)


def fig_fa(plt, fa_overall, fa_pos, fig_dir, cap=1.0):
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    knobs = list(fa_overall.keys())
    x = np.arange(len(knobs))
    w = 0.36
    for j, (split, col, hatch) in enumerate((("tune", AQUA, None), ("test", ORANGE, None))):
        v = np.array([fa_overall[k][split][0] for k in knobs])
        c = np.array([fa_overall[k][split][1:] for k in knobs])
        axs[0].bar(x + (j - 0.5) * (w + 0.02), v, w, color=col, label=split, edgecolor="white", linewidth=1)
        axs[0].errorbar(x + (j - 0.5) * (w + 0.02), v, yerr=[v - c[:, 0], c[:, 1] - v], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    for j, split in enumerate(("tune", "test")):
        for xx, k in zip(x, knobs):
            if fa_overall[k][split][0] == 0:
                axs[0].text(xx + (j - 0.5) * (w + 0.02), 0.04, "0", fontsize=7, color=INK2, ha="center", va="bottom")
    axs[0].axhline(cap, color=INK2, ls="--", lw=0.8)
    axs[0].text(len(knobs) - 0.55, 3.6, "dashed: FA cap,\napplied on tune", fontsize=7, color=INK2, ha="right", va="bottom")
    axs[0].set_xticks(x, [k.replace("CUSUM ", "CUSUM\n").replace(" (t1.0)", "\nchosen").replace(" (t0.5)", "\ntarget 0.5*").replace("3σ m=3", "3σ m=3\nchosen").replace("3σ m=2", "3σ m=2*") for k in knobs], fontsize=7.5)
    axs[0].set_title("overall (* test scored post hoc)", fontsize=8.5, color=INK2, loc="left")
    axs[0].set_ylabel("false alarms per hour")
    axs[0].legend(loc="upper left", fontsize=8)
    axs[0].grid(axis="x", visible=False)
    bins = [b[0] for b in POS_BINS]
    x = np.arange(len(bins))
    for j, (split, col) in enumerate((("tune", AQUA), ("test", ORANGE))):
        v = np.array(fa_pos[split]["fa_per_hour"])
        c = np.array(fa_pos[split]["ci"])
        axs[1].bar(x + (j - 0.5) * (w + 0.02), v, w, color=col, label=split, edgecolor="white", linewidth=1)
        axs[1].errorbar(x + (j - 0.5) * (w + 0.02), v, yerr=[v - c[:, 0], c[:, 1] - v], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    for j, split in enumerate(("tune", "test")):
        for xx, (v_, h_) in enumerate(zip(fa_pos[split]["fa_per_hour"], fa_pos[split]["hours"])):
            axs[1].text(xx + (j - 0.5) * (w + 0.02), -0.05, f"{h_:.1f} h", fontsize=6.5, color=INK2, ha="center", va="top")
    axs[1].set_ylim(bottom=-0.6)
    axs[1].set_xticks(x, [b.replace(" after an offset", "\nafter offset").replace("before the first onset", "before first\nonset") for b in bins], fontsize=8)
    axs[1].grid(axis="x", visible=False)
    axs[1].set_title(f"CUSUM h = {fa_pos['h']:.2f}, by time since the last switch back to nominal", fontsize=8.5, color=INK2, loc="left")
    fig.suptitle("False alarms, tune (seeds 0–5) vs test (seeds 6–15), 95% schedule-cluster CIs. Right: POST HOC; numbers under bars = robot-hours scored", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, fig_dir, "pathB_fa_tune_test")
    plt.close(fig)


def fig_adapt(plt, rep, fleet, fig_dir):
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 2.8))
    ax = axs[0]
    rows = [("replay − frozen", rep["replay_minus_frozen"]), ("onpolicy − frozen", rep["onpolicy_minus_frozen"]), ("replay − onpolicy", [rep["replay_minus_onpolicy"], *rep["replay_minus_onpolicy_ci"]])]
    y = np.arange(len(rows))[::-1]
    for yy, (n, v) in zip(y, rows):
        ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=BLUE, ecolor=BLUE, ms=6, capsize=3, lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(rep["bar"], color=INK2, lw=0.8, ls="--")
    ax.text(rep["bar"] - 0.003, y[-1] - 0.45, "pass bar\n+0.148", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("success rate difference, 50/60-min mean")
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_title("replay r3: 5200 policy, motor 0.6, k = 1, 60 min, 12 deployments", fontsize=8, color=INK2, loc="left")
    ax = axs[1]
    y = np.arange(len(fleet))[::-1]
    for yy, s in zip(y, fleet):
        v = [100 * q for q in s["gap_closed"]]
        ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=BLUE, ecolor=BLUE, ms=6, capsize=3, lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(50, color=INK2, lw=0.8, ls="--")
    ax.text(49, y[-1] - 0.45, "T50 target\n(50%)", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_yticks(y, [f"fleet seed {s['seed']}" for s in fleet])
    ax.set_xlabel("gap closed vs frozen control, whole 20 min (%)")
    ax.set_ylim(-0.6, len(fleet) - 0.3)
    ax.set_title("k = 10 fleet: RMA π₀, motor 0.5, 20 min, 10 paired cells", fontsize=8, color=INK2, loc="left")
    fig.suptitle("Online adaptation did not beat the frozen policy (95% bootstrap CIs over deployments / cells)", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, fig_dir, "adaptation_failure_summary")
    plt.close(fig)


# ---- main ---------------------------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--score", required=True, help="the pre-registered score.json (sigma, chosen knobs, primary numbers)")
    p.add_argument("--out", required=True)
    p.add_argument("--fig_dir", required=True)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--replay_verdict", default=None)
    p.add_argument("--fleet_jobs", nargs="*", default=[])
    a = p.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    prim = json.load(open(a.score))
    sigma, settle = prim["sigma_log"], prim["settle_s"]
    d = load_chain(a.jobs, CYCLE_S)
    st = world_state(d)
    z, _ = standardize(d["residual"], sigma, prim["baseline_cycles"])
    t, seed_of = d["t"], d["seed"]
    tune, test = np.where(d["split"] == "tune")[0], np.where(d["split"] == "test")[0]
    rng = np.random.default_rng(0)
    seeds = {"tune": sorted(set(seed_of[tune].tolist())), "test": sorted(set(seed_of[test].tolist()))}
    W = {s: seed_weights(len(seeds[s]), a.boot, rng) for s in seeds}
    robots = {"tune": tune, "test": test}
    res = {"label": "POST HOC, EXPLORATORY (supervisor 11-12); the primary result is score.json", "sigma_log": sigma}

    # CUSUM h for each target, calibrated on tune exactly as the scorer does
    h_grid = np.concatenate([np.arange(0.5, 10.0, 0.25), np.arange(10.0, 200.0, 1.0)])
    al_h, rs_h = alarms_cusum(z[:, tune], prim["cusum_k"], h_grid)
    rates = np.array([fa_rate(al_h[q], st[:, tune], z[:, tune], np.arange(len(tune)), CYCLE_S, settle, rs_h[q]) for q in range(len(h_grid))])
    del al_h, rs_h
    h_of = {tg: float(h_grid[np.where(rates <= tg)[0][0]]) for tg in (0.5, 1.0, 2.0)}
    assert abs(h_of[1.0] - prim["chosen"]["cusum"]["h"]) < 1e-9, (h_of, prim["chosen"]["cusum"])
    res["cusum_h_of_target"] = h_of
    knobs = {"3σ m=3": ("3s", 3), "3σ m=2": ("3s", 2), "CUSUM h=%.2f (t1.0)" % h_of[1.0]: ("cu", h_of[1.0]), "CUSUM h=%.2f (t0.5)" % h_of[0.5]: ("cu", h_of[0.5])}
    alarms, scores = {}, {}
    for name, (kind, v) in knobs.items():
        if kind == "3s":
            al, rs = alarms_three_sigma(z, prim["three_sigma_k"], v)
        else:
            al, rs = alarms_cusum(z, prim["cusum_k"], [v])
            al, rs = al[0], rs[0]
        alarms[name] = (al, rs)
        scores[name] = score(al, st, z, t, CYCLE_S, settle, rs)
    # (a) secondary table + reproduction check of the primary numbers
    sec = {}
    for name in knobs:
        sec[name] = {s: onset_metrics(scores[name], robots[s], seed_of, seeds[s], W[s]) for s in ("tune", "test")}
    chk = []
    for nm, det in (("3σ m=3", "three_sigma"), ("CUSUM h=%.2f (t1.0)" % h_of[1.0], "cusum")):
        pt = prim["test"][det]
        mine = sec[nm]["test"]
        chk.append(abs(pt["fa_per_hour"] - mine["fa_per_hour"][0]) < 1e-9 and abs(pt["censored_mean_delay_s_types"] - mine["type_mean_s"][0]) < 1e-9 and all(abs(pt[f]["miss_rate"] - mine[f]["miss_rate"][0]) < 1e-9 for f in FACTORS))
    res["reproduces_primary_point_estimates"] = bool(all(chk))
    # paired CUSUM - 3σ without friction, test (chosen knobs)
    for split in ("test",):
        b = sec["CUSUM h=%.2f (t1.0)" % h_of[1.0]][split]["_boot_type_mean_no_friction"] - sec["3σ m=3"][split]["_boot_type_mean_no_friction"]
        pt_ = sec["CUSUM h=%.2f (t1.0)" % h_of[1.0]][split]["type_mean_no_friction_s"][0] - sec["3σ m=3"][split]["type_mean_no_friction_s"][0]
        res["cusum_minus_3s_no_friction_test"] = [float(pt_), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    res["secondary"] = {k: {s: {kk: vv for kk, vv in m.items() if not kk.startswith("_")} for s, m in v.items()} for k, v in sec.items()}
    # (b) z traces and on-segment z
    tr = z_traces(z, st, test, seed_of, seeds["test"], W["test"])
    res["on_segment_z_test"] = on_segment_z(z, st, test, seed_of, seeds["test"], W["test"])
    res["on_segment_z_tune"] = on_segment_z(z, st, tune, seed_of, seeds["tune"], W["tune"])
    pk = {}
    for (al_, f), v in tr.items():
        if al_ == "onset":
            m = v["t_s"] > 0
            pk[f] = {"mean_z_0_10s": float(np.nanmean(v["mean"][(v["t_s"] > 0) & (v["t_s"] <= 10)])), "mean_z_10_35s": float(np.nanmean(v["mean"][(v["t_s"] > 10) & (v["t_s"] <= 35)])), "peak_mean_z": float(np.nanmax(v["mean"][m])), "t_peak_s": float(v["t_s"][m][np.nanargmax(v["mean"][m])])}
        else:
            m = v["t_s"] > 0
            pk[f]["after_offset_mean_z_0_10s"] = float(np.nanmean(v["mean"][(v["t_s"] > 0) & (v["t_s"] <= 10)]))
            pk[f]["after_offset_mean_z_10_30s"] = float(np.nanmean(v["mean"][(v["t_s"] > 10) & (v["t_s"] <= 30)]))
    res["z_trace_summary_test"] = pk
    # (c) FA by position, all four knobs, tune vs test, plus concentration
    fa_pos, fa_overall = {}, {}
    for name in knobs:
        al, rs = alarms[name]
        fa, hrs = fa_by_position(al, rs, st, z, settle)
        assert np.allclose(fa.sum(1), scores[name]["fa"]) and np.allclose(hrs.sum(1), scores[name]["hours"]), name
        fa_pos[name] = {}
        for s in ("tune", "test"):
            rb = robots[s]
            num = np.stack([fa[rb][seed_of[rb] == q].sum(0) for q in seeds[s]])
            den = np.stack([hrs[rb][seed_of[rb] == q].sum(0) for q in seeds[s]])
            pt_, lo, hi, _ = ratio_ci(num, den, W[s])
            per_seed = (num.sum(1) / den.sum(1)).round(2).tolist()
            fr = fa[rb].sum(1)
            top = np.sort(fr)[::-1]
            fa_pos[name][s] = {"fa_per_hour": pt_.tolist(), "ci": np.stack([lo, hi], 1).tolist(), "hours": den.sum(0).round(2).tolist(), "alarms": num.sum(0).astype(int).tolist(), "per_schedule_fa_per_hour": dict(zip(map(str, seeds[s]), per_seed)),
                               "robots_with_fa": int((fr > 0).sum()), "n_robots": int(len(rb)), "share_of_fa_top3_robots": float(top[:3].sum() / max(top.sum(), 1))}
        fa_overall[name] = {s: sec[name][s]["fa_per_hour"] for s in ("tune", "test")}
    res["fa_by_position"] = fa_pos
    res["fa_position_bins"] = [b[0] for b in POS_BINS]
    # nominal z level per split (scored nominal cycles), a drift check
    res["nominal_mean_z"] = {"tune": res["on_segment_z_tune"]["nominal"]["mean_z"], "test": res["on_segment_z_test"]["nominal"]["mean_z"]}
    # adaptation-failure evidence
    rep = replay_summary(a.replay_verdict, rng, a.boot) if a.replay_verdict else None
    fleet = fleet_summary(a.fleet_jobs, rng, a.boot) if a.fleet_jobs else None
    res["replay_r3"], res["fleet_k10"] = rep, fleet
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=float)
    # figures
    plt = setup_mpl()
    fig_table(plt, prim, a.fig_dir)
    fig_delay_miss(plt, prim, a.fig_dir)
    fig_z(plt, tr, a.fig_dir)
    cu = "CUSUM h=%.2f (t1.0)" % h_of[1.0]
    fig_fa(plt, fa_overall, {"h": h_of[1.0], "tune": fa_pos[cu]["tune"], "test": fa_pos[cu]["test"]}, a.fig_dir)
    if rep and fleet:
        fig_adapt(plt, rep, fleet, a.fig_dir)
    print(json.dumps({k: v for k, v in res.items() if k not in ("secondary", "fa_by_position")}, indent=1, default=float))


if __name__ == "__main__":
    main()

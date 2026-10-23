# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize deployment runs (decisions.md E2 / E4, E2 as updated 2026-10-21): one markdown table across arms x
pi_0 seeds x schedules with bootstrap CIs over deployments. Each arm gets raw whole-deployment scores: forward
progress is the ranking metric, summed reward is reported alongside and the two rankings are compared (a flag when
they disagree); the oracle is one arm in the table. Quiet-stretch progress is compared with the never arm, matched
on (checkpoint, schedule seed, env index). The hindsight-best schedule is the scheduled(T) period with the highest
mean progress (a post-hoc pick, not a run).

R2: schedules are tagged tune / test (ScheduleCfg.tune_seeds / test_seeds). Each arm family's knob (T, H, h, m) is
chosen on tune deployments only (highest mean progress); the hypotheses are reported on test deployments with
paired differences (same checkpoint, schedule, env) and bootstrap CIs. R7: bootstrap resamples pi_0 checkpoints
(seeds) first, then deployments within; the number of updater seeds per (pi_0, deployment) is reported.
No Isaac Sim needed.

  python tools/deploy_summarize.py --root /workspace/isaaclab/logs/deploy --out summary.md
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.metrics import bootstrap_ci, deployment_metrics, fmt_ci  # noqa: E402
from hexapod.deploy.schedule import Schedule  # noqa: E402
from hexapod.deploy.streams import StreamLog  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, help="directory of deploy_run.py outputs (searched recursively)")
parser.add_argument("--out", default=None, help="markdown output (default <root>/summary.md)")
parser.add_argument("--detect_window_min", type=float, default=None)
parser.add_argument("--bootstrap_n", type=int, default=1000)
parser.add_argument("--label_by_tag", action="store_true", help="append the run tag to the arm label (pilot comparisons of updater settings; off for the main study, where the label is the arm and its knob only)")
args = parser.parse_args()
mcfg = C.MetricsCfg(detect_window_min=args.detect_window_min, bootstrap_n=args.bootstrap_n)
scfg = C.ScheduleCfg()

rows = []  # one per deployment
for meta_path in sorted(glob.glob(os.path.join(args.root, "**", "meta.json"), recursive=True)):
    run_dir = os.path.dirname(meta_path)
    if not os.path.exists(os.path.join(run_dir, "streams.npz")):
        continue
    arrays, meta = StreamLog.load(run_dir)
    if "arm" not in meta or "env_schedule_seed" not in meta:
        continue
    budget = (meta.get("arm_cfg") or {}).get("budget_min") or (meta.get("updater_cfg") or {}).get("budget_min") or 1.0
    if budget == float("inf") or (isinstance(budget, str)):
        budget = (meta.get("updater_cfg") or {}).get("budget_min") or 1.0
    scheds = {}
    for i, seed in enumerate(meta["env_schedule_seed"]):
        if seed is None:
            s = None
        else:
            if seed not in scheds:
                scheds[seed] = Schedule.load(os.path.join(run_dir, f"schedule_seed{seed}.json"))
            s = scheds[seed]
        m = deployment_metrics(arrays, i, s, mcfg, budget)
        if not m:
            continue
        split = (meta.get("schedule_split") or [None] * len(meta["env_schedule_seed"]))[i]
        if split is None and seed is not None:
            split = "tune" if seed in scfg.tune_seeds else "test" if seed in scfg.test_seeds else "other"
        arm_i = (meta.get("arm_of_env") or [meta["arm"]] * len(meta["env_schedule_seed"]))[i]
        lab = (meta.get("arm_label_of_env") or [meta.get("arm_label") or meta["arm"]] * len(meta["env_schedule_seed"]))[i]
        tag = meta.get("tag")
        if tag is None:  # runs written before the tag was stored: <stamp>_<arm>_<tag>
            base = os.path.basename(run_dir.rstrip("/"))
            tag = base.split(f"_{meta['arm']}_", 1)[1] if f"_{meta['arm']}_" in base else ""
        if args.label_by_tag and tag:
            lab = f"{lab}[{tag}]"
        rep_i = (meta.get("replicate_of_env") or list(range(len(meta["env_schedule_seed"]))))[i]
        m.update({"arm": arm_i, "arm_label": lab, "replicate": rep_i, "checkpoint": os.path.basename(os.path.dirname(meta["checkpoint"])) + "/" + os.path.basename(meta["checkpoint"]), "schedule_seed": seed, "split": split or "none", "env": i, "env_seed": (meta.get("deploy_cfg") or {}).get("env_seed", 0), "period_min": (meta.get("arm_cfg") or {}).get("period_min"), "run_dir": run_dir})
        rows.append(m)
print(f"[summarize] {len(rows)} deployments from {len({r['run_dir'] for r in rows})} runs")


def key(r):
    """Pairing key across arms: same pi_0, schedule, updater seed and replicate (= terrain cell); in single-arm runs the
    replicate is the env index, in multi-arm runs env // number of arms."""
    return (r["checkpoint"], r["schedule_seed"], r["env_seed"], r["replicate"])


never = {key(r): r for r in rows if r["arm"] == "never"}
for r in rows:
    k = key(r)
    r["quiet_vs_never_per_h"] = r["quiet_progress_per_h"] - never[k]["quiet_progress_per_h"] if k in never else float("nan")

arm_label = lambda r: r["arm_label"]
groups: dict[str, list] = {}
for r in rows:
    groups.setdefault(arm_label(r), []).append(r)
# hindsight-best schedule: the scheduled(T) with the highest mean progress (over all deployments; a post-hoc pick)
sched_groups = {g: v for g, v in groups.items() if g.startswith("scheduled(")}
if sched_groups:
    best = max(sched_groups, key=lambda g: np.mean([r["progress_m"] for r in sched_groups[g]]))
    groups[f"hindsight_best_schedule[{best}]"] = sched_groups[best]

# ---- R2: knob selection on tune, hypotheses on test -----------------------------------------------------------
tune = [r for r in rows if r["split"] == "tune"]
test = [r for r in rows if r["split"] == "test"]
selected: dict[str, str] = {}  # arm family -> chosen label
for fam in ("scheduled", "periodic_reset", "pause_hold", "three_sigma", "cusum"):
    labels = {r["arm_label"] for r in tune if r["arm"] == fam}
    if labels:
        selected[fam] = max(labels, key=lambda g: np.mean([r["progress_m"] for r in tune if r["arm_label"] == g]))
for fam in ("never", "always", "oracle"):
    if any(r["arm"] == fam for r in rows):
        selected[fam] = fam


def paired_diff(label_a: str, label_b: str, pool: list, metric: str = "progress_m"):
    """Per-deployment differences a - b matched on (checkpoint, schedule, env_seed, env); seed-then-deployment bootstrap."""
    ka = {key(r): r[metric] for r in pool if r["arm_label"] == label_a}
    kb = {key(r): r[metric] for r in pool if r["arm_label"] == label_b}
    common = sorted(set(ka) & set(kb))
    if not common:
        return None
    diffs = np.array([ka[k] - kb[k] for k in common])
    cks = np.array([k[0] for k in common])
    lo, hi = hierarchical_bootstrap_ci(diffs, cks, args.bootstrap_n)
    return {"n": len(common), "mean": float(diffs.mean()), "ci": [lo, hi], "n_checkpoints": len(set(cks))}


def hierarchical_bootstrap_ci(x, groups_arr, n_boot, ci=0.95, rng=None):
    """R7: resample pi_0 checkpoints with replacement, then deployments within each resampled checkpoint."""
    rng = rng or np.random.default_rng(0)
    uniq = np.unique(groups_arr)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        vals = []
        for g in pick:
            xg = x[groups_arr == g]
            vals.append(xg[rng.integers(0, len(xg), len(xg))])
        means.append(np.concatenate(vals).mean())
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


hyp = []
trig = [selected[f] for f in ("cusum", "three_sigma") if f in selected]
if trig and "scheduled" in selected:
    best_trig = max(trig, key=lambda g: np.mean([r["progress_m"] for r in tune if r["arm_label"] == g]))
    d = paired_diff(best_trig, selected["scheduled"], test)
    hyp.append(("H1: best trigger - best schedule (test, progress [m])", best_trig, selected["scheduled"], d, "holds (CI excludes 0, positive)" if d and d["ci"][0] > 0 else "fails" if d else "no paired test deployments"))
if "always" in selected and "never" in selected:
    d = paired_diff("always", "never", test, "quiet_progress_m")
    hyp.append(("H2: always - never on quiet stretches (test, progress [m])", "always", "never", d, "holds (CI excludes 0, negative = cost)" if d and d["ci"][1] < 0 else "fails" if d else "no paired test deployments"))
# R7: updater seeds per (pi_0, deployment)
seed_counts = {}
for r in rows:
    seed_counts.setdefault((r["arm_label"], r["checkpoint"], r["schedule_seed"], r["replicate"]), set()).add(r["env_seed"])
n_upd_seeds = sorted({len(v) for v in seed_counts.values()}) if seed_counts else []

cols = [("progress_m", "progress [m]"), ("reward_sum", "summed reward"), ("falls_per_h", "falls/h"), ("track_err", "tracking err [m/s]"), ("false_alarms_per_h", "false alarms/h"), ("detection_delay_s", "detection delay [s]"), ("missed", "missed changes"), ("quiet_vs_never_per_h", "quiet progress vs never [m/h]"), ("recurrence_gain_m", "recurrence gain [m]"), ("adapt_min", "adapt time [min]")]
seg_cols = [("quiet_progress_m", "quiet [m]"), ("quiet_min", "quiet [min]"), ("post_change_progress_m", "post-change [m]"), ("post_change_min", "post-change [min]"), ("recurrence_progress_m", "recurrence [m]"), ("recurrence_min", "recurrence [min]")]
order = ["never", "always", "scheduled", "periodic_reset", "pause_hold", "three_sigma", "cusum", "triggered", "oracle", "hindsight"]
names = sorted(groups, key=lambda g: next((i for i, o in enumerate(order) if g.startswith(o)), 99))
rank_p = sorted(names, key=lambda g: -np.mean([r["progress_m"] for r in groups[g]]))
rank_r = sorted(names, key=lambda g: -np.nanmean([r["reward_sum"] for r in groups[g]]))
md = [f"# Deployment summary ({len(rows)} deployments, {len({r['checkpoint'] for r in rows})} pi_0 checkpoints, {len({r['schedule_seed'] for r in rows})} schedules)", "",
      "mean [95% bootstrap CI over deployments]; ranking metric = forward progress; quiet progress vs never is matched per (checkpoint, schedule, env).", "",
      f"ranking by progress: {' > '.join(rank_p)}", f"ranking by summed reward: {' > '.join(rank_r)}",
      ("**rankings agree**" if rank_p == rank_r else "**rankings disagree** (progress is the primary metric, E2)"), "",
      f"R2 split: {len(tune)} tune / {len(test)} test / {len(rows) - len(tune) - len(test)} untagged deployments; knobs chosen on tune: " + ", ".join(f"{k} -> {v}" for k, v in selected.items() if k != v), "",
      f"R7: pi_0 checkpoints {len({r['checkpoint'] for r in rows})}; updater seeds per (arm, pi_0, schedule, env): {n_upd_seeds}", "",
      "| arm | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
for g in names:
    rs = groups[g]
    md.append(f"| {g} | {len(rs)} | " + " | ".join(fmt_ci([r[c] for r in rs], args.bootstrap_n) for c, _ in cols) + " |")
md += ["", "## Hypotheses on held-out (test) deployments, paired differences, seed-then-deployment bootstrap", ""]
if hyp:
    md += ["| hypothesis | a | b | n pairs (pi_0 seeds) | mean a - b | 95% CI | verdict |", "|---|---|---|---|---|---|---|"]
    for name, a, b, d, verdict in hyp:
        md.append(f"| {name} | {a} | {b} | " + (f"{d['n']} ({d['n_checkpoints']}) | {d['mean']:.2f} | [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] | {verdict} |" if d else "– | – | – | no paired test deployments |"))
else:
    md.append("(no tune/test-tagged runs yet)")
md += ["", "## Progress decomposition (quiet / post-change / recurrence; window = adaptation budget)", "", "| arm | " + " | ".join(c[1] for c in seg_cols) + " |", "|---|" + "---|" * len(seg_cols)]
for g in names:
    rs = groups[g]
    md.append(f"| {g} | " + " | ".join(fmt_ci([r[c] for r in rs], args.bootstrap_n) for c, _ in seg_cols) + " |")
md += ["", "## Per checkpoint x schedule (progress [m], mean over envs)", ""]
cks = sorted({r["checkpoint"] for r in rows})
seeds = sorted({str(r["schedule_seed"]) for r in rows})
md.append("| arm | checkpoint | " + " | ".join(f"sched {s}" for s in seeds) + " |")
md.append("|---|---|" + "---|" * len(seeds))
for g in names:
    for ck in cks:
        vals = []
        for s in seeds:
            v = [r["progress_m"] for r in groups[g] if r["checkpoint"] == ck and str(r["schedule_seed"]) == s]
            vals.append(f"{np.mean(v):.1f}" if v else "–")
        if any(v != "–" for v in vals):
            md.append(f"| {g} | {ck} | " + " | ".join(vals) + " |")
out = args.out or os.path.join(args.root, "summary.md")
open(out, "w").write("\n".join(md) + "\n")
json.dump(rows, open(os.path.splitext(out)[0] + ".json", "w"), indent=1, default=float)
print("\n".join(md))
print("[summarize] wrote", out)

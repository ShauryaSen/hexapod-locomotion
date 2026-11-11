# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path B offline detector scoring (preregistration 3 and 6). Replays the 3-sigma rule and CUSUM on the per-cycle
residuals that frozen (Never-arm) deployments logged, picks each detector's knob on the tune schedules only, and
scores the test schedules with the knob frozen. CPU and numpy only.

Per robot (one robot = one deployment trace; its schedule seed and split come from meta.json):
  - x = log residual per gait cycle. The first ``baseline_cycles`` valid cycles set the robot's own mean (R5); no
    alarms there. z = (x - mu_robot) / sigma, sigma = the ensemble's global nominal sd of the log residual.
  - Detectors as hexapod/deploy/detectors.py: 3-sigma = z > 3 for m cycles in a row; CUSUM S = max(0, S + z - k),
    alarm when S > h. Both restart after an alarm. NaN z (falls, stand-ups, baseline) = no evidence.
  - The true world state per cycle comes from the logged payload_kg / motor_strength / friction streams. An onset is
    the first cycle with a test value on; its time is that cycle's start (0.4 s resolution, delays are overstated by
    at most one cycle).
  - Detection delay = end of the first alarming cycle inside the on-segment - onset. No alarm before the value
    switches back to nominal = miss (the window is the on-segment). Censored delay = delay, or the on-segment's
    length for a miss.
  - False alarms = alarms on nominal cycles, outside the baseline and outside ``settle_s`` after a switch back to
    nominal; per hour of valid-z nominal time in that set. Alarms in the settle windows and repeat alarms inside an
    on-segment are counted separately.

Knob choice (tune split only): 3-sigma m in {2, 3, 5}; CUSUM h calibrated on the tune nominal time to each target
false-alarm rate in {0.5, 1, 2} per hour. Per detector, the knob with the lowest tune censored delay (mean over the
three change types, each type weighted equally) among the knobs whose tune false-alarm rate is <= ``fa_cap``; if none
qualifies, the lowest false-alarm rate.

Uncertainty: 95% bootstrap CIs over schedules (robots of one schedule move together), 10 000 resamples.

  python tools/deploy_score_detectors.py --jobs <job_00> <job_01> ... --ensemble <ensemble.pt> --out <prefix>
  python tools/deploy_score_detectors.py --selftest     # synthetic changes on known answers
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

FACTORS = ("payload", "motor", "friction")


# ---- loading ----------------------------------------------------------------------------------------------------


def load_chain(jobs: list[str], cycle_s: float = 0.4) -> dict:
    """Concatenate chained jobs of one process along cycles; checks the robot layout and that time continues."""
    keys = ("t_s", "residual", "payload_kg", "motor_strength", "friction")
    parts = {k: [] for k in keys}
    meta0 = None
    t_prev = None
    for j in jobs:
        z = np.load(os.path.join(j, "streams.npz"))
        m = json.load(open(os.path.join(j, "meta.json")))
        if meta0 is None:
            meta0 = m
        elif m.get("env_schedule_seed") != meta0.get("env_schedule_seed"):
            raise ValueError(f"{j}: schedule layout differs from {jobs[0]}")
        t = z["t_s"][:, 0]
        if t_prev is not None and abs(t[0] - (t_prev + cycle_s)) > 1e-3:
            raise ValueError(f"{j}: starts at t = {t[0]:.2f} s, previous job ended at {t_prev:.2f} s (expected +{cycle_s} s)")
        t_prev = float(t[-1])
        for k in keys:
            parts[k].append(z[k])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["t"] = out.pop("t_s")[:, 0]
    n = out["residual"].shape[1]
    out["seed"] = np.array([s if s is not None else -1 for s in (meta0.get("env_schedule_seed") or [None] * n)])
    out["split"] = np.array([s or "none" for s in (meta0.get("schedule_split") or [None] * n)])
    return out


def world_state(d: dict) -> np.ndarray:
    """(cycles, robots) int: -1 nominal, else the index of the active factor in FACTORS. Raises on overlaps."""
    nom_fr = d["friction"][0]
    on = np.stack([d["payload_kg"] > 1e-6, d["motor_strength"] < 1 - 1e-6, d["friction"] < nom_fr[None, :] - 1e-6])
    if (on.sum(0) > 1).any():
        raise ValueError("two test values on at once: path B schedules never overlap")
    st = np.full(on.shape[1:], -1, dtype=int)
    for f in range(3):
        st[on[f]] = f
    return st


def segments(st_col: np.ndarray) -> list[tuple[int, int, int]]:
    """On-segments of one robot: (factor, first cycle, end cycle exclusive)."""
    out, c, n = [], 0, len(st_col)
    while c < n:
        if st_col[c] >= 0:
            e = c
            while e < n and st_col[e] == st_col[c]:
                e += 1
            out.append((int(st_col[c]), c, e))
            c = e
        else:
            c += 1
    return out


def standardize(res: np.ndarray, sigma: float, baseline_cycles: int) -> tuple[np.ndarray, np.ndarray]:
    """z per robot with its own baseline mean; also the cycle at which each robot's baseline ended."""
    x = np.log(np.clip(res, 1e-12, None))
    z = np.full_like(x, np.nan)
    base_end = np.full(x.shape[1], -1)
    for i in range(x.shape[1]):
        valid = np.where(~np.isnan(x[:, i]))[0]
        if len(valid) <= baseline_cycles:
            continue
        mu = x[valid[:baseline_cycles], i].mean()
        z[:, i] = (x[:, i] - mu) / sigma
        b = valid[baseline_cycles - 1] if baseline_cycles > 0 else -1
        z[: b + 1, i] = np.nan
        base_end[i] = b
    return z, base_end


# ---- detectors (vectorised over robots and knobs; same rules as hexapod/deploy/detectors.py) ---------------------


def alarms_three_sigma(z: np.ndarray, k: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Alarms (T, n) and, at each alarm, the cycle where its evidence run began (-1 elsewhere)."""
    T, n = z.shape
    c = np.zeros(n, dtype=int)
    start = np.zeros(n, dtype=int)
    out = np.zeros((T, n), dtype=bool)
    run_start = np.full((T, n), -1, dtype=np.int16)
    for t in range(T):
        v = z[t]
        valid = ~np.isnan(v)
        above = valid & (v > k)
        start = np.where(above & (c == 0), t, start)
        c = np.where(above, c + 1, np.where(valid, 0, c))
        a = c >= m
        run_start[t] = np.where(a, start, -1)
        c = np.where(a, 0, c)
        out[t] = a
    return out, run_start


def alarms_cusum(z: np.ndarray, k: float, hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(len(hs), T, n) alarms for every threshold at once, and at each alarm the cycle where S last left 0."""
    T, n = z.shape
    hs = np.asarray(hs, dtype=float)[:, None]
    S = np.zeros((len(hs), n))
    start = np.zeros((len(hs), n), dtype=int)
    out = np.zeros((len(hs), T, n), dtype=bool)
    run_start = np.full((len(hs), T, n), -1, dtype=np.int16)
    for t in range(T):
        v = z[t][None, :]
        valid = ~np.isnan(v)
        S_new = np.where(valid, np.maximum(0.0, S + np.nan_to_num(v) - k), S)
        start = np.where((S == 0) & (S_new > 0), t, start)
        S = S_new
        a = S > hs
        run_start[:, t] = np.where(a, start, -1)
        S = np.where(a, 0.0, S)
        out[:, t] = a
    return out, run_start


# ---- scoring ----------------------------------------------------------------------------------------------------


def score(alarm: np.ndarray, st: np.ndarray, z: np.ndarray, t: np.ndarray, cycle_s: float, settle_s: float, run_start: np.ndarray | None = None) -> dict:
    """Per-robot records: onsets (factor, delay or None, censored delay), false alarms and scored nominal hours.
    A nominal alarm whose evidence run began before the end of the last on-segment (``run_start``) is a carry-over
    alarm of that change, not a false alarm."""
    T, n = st.shape
    settle_c = int(round(settle_s / cycle_s))
    valid = ~np.isnan(z)
    onsets, fa, hours, settle_alarms, repeat_alarms, carry = [], np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        excl = np.zeros(T, dtype=bool)
        last_end = np.full(T, -1)
        for f, a, e in segments(st[:, i]):
            last_end[e:] = e
            t_on = t[a] - cycle_s
            hit = np.where(alarm[a:e, i])[0]
            seg_len = t[e - 1] - t_on
            if len(hit):
                dly = float(t[a + hit[0]] - t_on)
                onsets.append((i, f, dly, dly, seg_len))
                repeat_alarms[i] += len(hit) - 1
            else:
                onsets.append((i, f, None, float(seg_len), seg_len))
            excl[e : e + settle_c] = True
        nominal = (st[:, i] < 0) & valid[:, i]
        if run_start is not None:
            co = alarm[:, i] & nominal & (run_start[:, i] >= 0) & (run_start[:, i] < last_end)
            carry[i] = co.sum()
            nominal = nominal & ~co  # carry-over cycles are neither false alarms nor scored nominal time
        scored = nominal & ~excl
        fa[i] = alarm[scored, i].sum()
        settle_alarms[i] = alarm[nominal & excl, i].sum()
        hours[i] = scored.sum() * cycle_s / 3600.0
    return {"onsets": onsets, "fa": fa, "hours": hours, "settle_alarms": settle_alarms, "repeat_alarms": repeat_alarms, "carry_over_alarms": carry}


def summarize(sc: dict, robots: np.ndarray, seed_of: np.ndarray, rng, boot: int) -> dict:
    """Metrics over the given robots, with 95% CIs from a bootstrap over schedules."""
    rset = set(robots.tolist())
    on = [o for o in sc["onsets"] if o[0] in rset]
    seeds = sorted(set(seed_of[robots].tolist()))

    def metrics(seed_w: dict) -> dict:
        w_rob = {i: seed_w.get(seed_of[i], 0) for i in robots}
        out = {}
        hrs = sum(sc["hours"][i] * w_rob[i] for i in robots)
        out["fa_per_hour"] = sum(sc["fa"][i] * w_rob[i] for i in robots) / max(hrs, 1e-12)
        cens_types = []
        for f, name in enumerate(FACTORS):
            rows = [(o, w_rob[o[0]]) for o in on if o[1] == f]
            wsum = sum(w for _, w in rows)
            if wsum == 0:
                out[name] = {"n": 0}
                continue
            miss = sum(w for o, w in rows if o[2] is None) / wsum
            det = [(o[2], w) for o, w in rows if o[2] is not None]
            cens = sum(o[3] * w for o, w in rows) / wsum
            delays = np.repeat([d for d, _ in det], [w for _, w in det]) if det else np.array([])
            out[name] = {"n": int(wsum), "miss_rate": miss, "mean_delay_detected_s": float(delays.mean()) if len(delays) else None, "median_delay_detected_s": float(np.median(delays)) if len(delays) else None, "censored_mean_delay_s": cens}
            cens_types.append(cens)
        out["censored_mean_delay_s_types"] = float(np.mean(cens_types)) if cens_types else None
        return out

    point = metrics({s: 1 for s in seeds})
    if boot and len(seeds) > 1:
        draws = []
        for _ in range(boot):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            w = {}
            for s in pick:
                w[s] = w.get(s, 0) + 1
            draws.append(metrics(w))

        def ci(get):
            v = np.array([x for x in (get(d) for d in draws) if x is not None], dtype=float)
            return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None

        point["fa_per_hour_ci"] = ci(lambda d: d["fa_per_hour"])
        point["censored_mean_delay_s_types_ci"] = ci(lambda d: d["censored_mean_delay_s_types"])
        for name in FACTORS:
            if point[name].get("n"):
                point[name]["miss_rate_ci"] = ci(lambda d, nm=name: d[nm].get("miss_rate"))
                point[name]["censored_mean_delay_s_ci"] = ci(lambda d, nm=name: d[nm].get("censored_mean_delay_s"))
                point[name]["mean_delay_detected_s_ci"] = ci(lambda d, nm=name: d[nm].get("mean_delay_detected_s"))
    point["n_schedules"], point["n_robots"] = len(seeds), int(len(robots))
    point["robot_hours_scored_nominal"] = float(sc["hours"][robots].sum())
    point["settle_alarms"], point["repeat_alarms"], point["carry_over_alarms"] = int(sc["settle_alarms"][robots].sum()), int(sc["repeat_alarms"][robots].sum()), int(sc["carry_over_alarms"][robots].sum())
    return point


def fa_rate(alarm: np.ndarray, st: np.ndarray, z: np.ndarray, robots: np.ndarray, cycle_s: float, settle_s: float, run_start: np.ndarray) -> float:
    sc = score(alarm[:, robots], st[:, robots], z[:, robots], np.arange(1, st.shape[0] + 1) * cycle_s, cycle_s, settle_s, run_start[:, robots])
    return float(sc["fa"].sum() / max(sc["hours"].sum(), 1e-12))


def run(d: dict, sigma: float, a) -> dict:
    cycle_s = a.cycle_s
    st = world_state(d)
    z, base_end = standardize(d["residual"], sigma, a.baseline_cycles)
    t = d["t"]
    seed_of = d["seed"]
    tune = np.where(d["split"] == "tune")[0]
    test = np.where(d["split"] == "test")[0]
    first_on = np.array([next((a_ for f, a_, e in segments(st[:, i])), st.shape[0]) for i in range(st.shape[1])])
    late = int(((base_end >= first_on) | (base_end < 0)).sum())
    rng = np.random.default_rng(0)
    res = {"sigma_log": sigma, "baseline_cycles": a.baseline_cycles, "settle_s": a.settle_s, "fa_cap": a.fa_cap, "cusum_k": a.cusum_k, "three_sigma_k": a.three_sigma_k,
           "cycles": int(st.shape[0]), "robots": int(st.shape[1]), "robots_tune": int(len(tune)), "robots_test": int(len(test)),
           "robots_baseline_not_done_before_first_onset": late, "knobs": {}, "chosen": {}, "test": {}, "tune": {}}
    # tune: every knob's tune metrics
    cand = {}
    for m in a.three_sigma_m:
        al, rs = alarms_three_sigma(z, a.three_sigma_k, m)
        cand[f"three_sigma:m={m}"] = ("three_sigma", (al, rs), {"m": m})
    h_grid = np.concatenate([np.arange(0.5, 10.0, 0.25), np.arange(10.0, 200.0, 1.0)])
    if len(tune):
        al_h, rs_h = alarms_cusum(z[:, tune], a.cusum_k, h_grid)
        rates = np.array([fa_rate(al_h[q], st[:, tune], z[:, tune], np.arange(len(tune)), cycle_s, a.settle_s, rs_h[q]) for q in range(len(h_grid))])
        del al_h, rs_h
        res["cusum_tune_grid"] = [(float(h), float(r)) for h, r in zip(h_grid, rates) if r > 0][:40]
        for target in a.cusum_targets:
            ok = np.where(rates <= target)[0]
            h = float(h_grid[ok[0]]) if len(ok) else float(h_grid[-1])
            al, rs = alarms_cusum(z, a.cusum_k, [h])
            cand[f"cusum:target={target}"] = ("cusum", (al[0], rs[0]), {"target_fa_per_hour": target, "h": h})
    else:
        for h in a.cusum_h:
            al, rs = alarms_cusum(z, a.cusum_k, [h])
            cand[f"cusum:h={h}"] = ("cusum", (al[0], rs[0]), {"h": h})
    scores = {name: score(ar[0], st, z, t, cycle_s, a.settle_s, ar[1]) for name, (_, ar, _) in cand.items()}
    for name, (det, _, knob) in cand.items():
        res["knobs"][name] = {"detector": det, **knob}
        if len(tune):
            res["tune"][name] = summarize(scores[name], tune, seed_of, rng, a.boot_tune)
    # choice per detector, tune only
    for det in ("three_sigma", "cusum"):
        names = [nm for nm, (dd, _, _) in cand.items() if dd == det]
        if not len(tune):
            continue
        tm = {nm: res["tune"][nm] for nm in names}
        ok = [nm for nm in names if tm[nm]["fa_per_hour"] <= a.fa_cap and tm[nm]["censored_mean_delay_s_types"] is not None]
        pick = min(ok, key=lambda nm: tm[nm]["censored_mean_delay_s_types"]) if ok else min(names, key=lambda nm: tm[nm]["fa_per_hour"])
        res["chosen"][det] = {"knob": pick, **res["knobs"][pick], "rule": "lowest tune censored delay with tune FA/h <= cap" if ok else "no knob under the cap: lowest tune FA/h"}
    # test, knobs frozen
    if len(test):
        for det, ch in res["chosen"].items():
            res["test"][det] = summarize(scores[ch["knob"]], test, seed_of, rng, a.boot)
        if len(res["chosen"]) == 2:
            s3, sc_ = scores[res["chosen"]["three_sigma"]["knob"]], scores[res["chosen"]["cusum"]["knob"]]
            seeds = sorted(set(seed_of[test].tolist()))

            def per_seed(sc):
                v = {}
                for s in seeds:
                    rs = set(np.where(seed_of == s)[0].tolist())
                    tps = [np.mean([o[3] for o in sc["onsets"] if o[0] in rs and o[1] == f]) for f in range(3) if any(o[0] in rs and o[1] == f for o in sc["onsets"])]
                    v[s] = float(np.mean(tps)) if tps else np.nan
                return v

            a3, ac = per_seed(s3), per_seed(sc_)
            diff = np.array([ac[s] - a3[s] for s in seeds])
            bs = diff[rng.integers(0, len(diff), size=(a.boot, len(diff)))].mean(1)
            res["test"]["cusum_minus_three_sigma_censored_delay_s"] = {"mean": float(diff.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "per_schedule": diff.round(2).tolist(), "label": "descriptive (path B has no pre-registered hypothesis between the detectors)"}
    if not len(tune) and not len(test):
        res["all"] = {name: summarize(scores[name], np.arange(st.shape[1]), seed_of, rng, a.boot) for name in cand}
    return res


def md_table(res: dict) -> str:
    lines = []

    def fmt(x, ci=None, p=1, pct=False):
        if x is None:
            return "-"
        s = f"{100 * x:.0f}%" if pct else f"{x:.{p}f}"
        if ci:
            s += f" [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]" if pct else f" [{ci[0]:.{p}f}, {ci[1]:.{p}f}]"
        return s

    for block in ("tune", "test", "all"):
        if not res.get(block):
            continue
        lines += [f"**{block}**", "", "| knob | FA/h | " + " | ".join(f"{f}: miss / censored delay s / delay if detected s" for f in FACTORS) + " | censored delay, type mean s |", "|---|---|" + "---|" * 4]
        for name, m in res[block].items():
            if not isinstance(m, dict) or "fa_per_hour" not in m:
                continue
            label = res["chosen"].get(name, {}).get("knob", name)
            cells = []
            for f in FACTORS:
                x = m.get(f, {})
                cells.append("n=0" if not x.get("n") else f"{fmt(x['miss_rate'], x.get('miss_rate_ci'), pct=True)} / {fmt(x['censored_mean_delay_s'], x.get('censored_mean_delay_s_ci'))} / {fmt(x['mean_delay_detected_s'], x.get('mean_delay_detected_s_ci'))}")
            lines.append(f"| {label} | {fmt(m['fa_per_hour'], m.get('fa_per_hour_ci'), 2)} | " + " | ".join(cells) + f" | {fmt(m['censored_mean_delay_s_types'], m.get('censored_mean_delay_s_types_ci'))} |")
        lines.append("")
    if res.get("chosen"):
        lines.append("chosen on tune: " + "; ".join(f"{d}: {c['knob']} ({c['rule']})" + (f", h = {c['h']}" if "h" in c else "") for d, c in res["chosen"].items()))
    if "cusum_minus_three_sigma_censored_delay_s" in res.get("test", {}):
        x = res["test"]["cusum_minus_three_sigma_censored_delay_s"]
        lines.append(f"test, CUSUM - 3-sigma censored delay (type mean, paired by schedule): {x['mean']:+.2f} s [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] ({x['label']})")
    return "\n".join(lines)


# ---- self-test --------------------------------------------------------------------------------------------------


def selftest(a) -> None:
    """Known answers on a synthetic chain: 4 schedules (2 tune, 2 test) x 3 robots, 600 cycles, sigma = 1."""
    T, n, cs = 600, 12, 0.4
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(T, n)) * 0.2  # quiet nominal z ~ N(0, 0.2)
    st_pay, st_mot, st_fr = np.zeros((T, n)), np.ones((T, n)), np.ones((T, n))
    # every robot: payload on cycles 200-299 (z + 5: 3-sigma m = 2 alarms at its 2nd cycle), motor 400-449 (z + 0.3: never above 3)
    x[200:300] += 5.0
    x[400:450] += 0.3
    st_pay[200:300] = 6.0
    st_mot[400:450] = 0.5
    x[520:522, 0] += 10.0  # one nominal false alarm for robot 0 (2 cycles above 3)
    x[300:302, 1] += 10.0  # inside the settle window after payload switches off: not a false alarm
    res = np.exp(x)
    d = {"t": np.arange(1, T + 1) * cs, "residual": res, "payload_kg": st_pay, "motor_strength": st_mot, "friction": st_fr,
         "seed": np.repeat(np.arange(4), 3), "split": np.array(["tune"] * 6 + ["test"] * 6)}
    b = argparse.Namespace(**{**vars(a), "baseline_cycles": 50, "three_sigma_m": [2], "cusum_targets": [1.0], "boot": 200, "boot_tune": 0})
    out = run(d, 1.0, b)
    al, rs = alarms_three_sigma(standardize(res, 1.0, 50)[0], 3.0, 2)
    sc = score(al, world_state(d), standardize(res, 1.0, 50)[0], d["t"], cs, 10.0, rs)
    pay = [o for o in sc["onsets"] if o[1] == 0]
    mot = [o for o in sc["onsets"] if o[1] == 1]
    checks = {
        "payload delay = 2 cycles from the onset cycle start (0.8 s) on every robot": all(abs(o[2] - 0.8) < 1e-6 for o in pay) and len(pay) == n,
        "motor (z + 0.3) missed by 3-sigma on every robot, censored at its 50-cycle segment (20.0 s)": all(o[2] is None and abs(o[3] - 20.0) < 1e-6 for o in mot),
        "exactly one false alarm (robot 0); the settle-window spike of robot 1 is not counted": sc["fa"].sum() == 1 and sc["fa"][0] == 1 and sc["settle_alarms"][1] == 1,
        "repeat alarms inside the payload segment are counted apart (100 cycles of z ~ 5: 50 per robot)": int(sc["repeat_alarms"][2]) == 49,
        "world state has no overlaps and 2 segments per robot": all(len(segments(world_state(d)[:, i])) == 2 for i in range(n)),
        "knob chosen on tune, test scored with it": "three_sigma" in out["chosen"] and out["test"]["three_sigma"]["n_robots"] == 6,
        "CUSUM h calibrated on tune, motor (z + 0.3 < k = 0.5) missed": out["test"]["cusum"]["motor"]["miss_rate"] == 1.0,
    }
    zz = standardize(res, 1.0, 50)[0]
    h = out["chosen"]["cusum"]["h"]
    exp = [0.4 * (1 + int(np.argmax(np.cumsum(np.maximum(zz[200:300, i] - 0.5, -1e9)) > h))) for i in range(6, 12)]  # S never hits 0 there
    alc, rsc = alarms_cusum(zz, 0.5, [h])
    got = [o[2] for o in score(alc[0], world_state(d), zz, d["t"], cs, 10.0, rsc[0])["onsets"] if o[1] == 0 and o[0] >= 6]
    zc = np.zeros((40, 1)); zc[5:15] = 2.0; zc[15:20] = 1.0  # S rises from cycle 5 to 15 (on-segment 5-14), alarm after it
    stc = np.full((40, 1), -1); stc[5:15] = 1
    alc2, rsc2 = alarms_cusum(zc, 0.5, [16.0])
    scc = score(alc2[0], stc, zc, np.arange(1, 41) * cs, cs, 0.0, rsc2[0])
    checks["CUSUM payload delay = first cycle with sum(z - k) > h, on every test robot"] = np.allclose(exp, got)
    checks["CUSUM evidence from a missed change alarming after it ends = carry-over, not a false alarm (S: 1.5 x 10 = 15, +0.5 x 5 -> 17.5 > 16 at cycle 18)"] = scc["onsets"][0][2] is None and scc["carry_over_alarms"][0] == 1 and scc["fa"][0] == 0
    for k_, v in checks.items():
        print(("PASS " if v else "FAIL ") + k_)
    print(md_table(out))
    if not all(checks.values()):
        raise SystemExit("selftest FAILED")
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", nargs="*", default=[], help="chained job dirs of one process, in time order (streams.npz + meta.json)")
    p.add_argument("--ensemble", default=None, help="ensemble.pt: sigma = its nominal sd of the log residual (needs torch)")
    p.add_argument("--sigma_log", type=float, default=None, help="override sigma")
    p.add_argument("--baseline_cycles", type=int, default=150, help="R5 per-deployment baseline (DetectorCfg.baseline_cycles)")
    p.add_argument("--cycle_s", type=float, default=0.4)
    p.add_argument("--settle_s", type=float, default=10.0, help="nominal time after a switch back to nominal that is not scored for false alarms")
    p.add_argument("--three_sigma_k", type=float, default=3.0)
    p.add_argument("--three_sigma_m", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--cusum_k", type=float, default=0.5)
    p.add_argument("--cusum_targets", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--cusum_h", type=float, nargs="+", default=[14.0], help="h values to score when there is no tune split")
    p.add_argument("--fa_cap", type=float, default=1.0, help="knob choice: tune FA/h must be <= this")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--boot_tune", type=int, default=1000)
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest(a)
        raise SystemExit(0)
    sigma = a.sigma_log
    if sigma is None:
        import torch

        ck = torch.load(a.ensemble, map_location="cpu", weights_only=False)
        sigma = float(ck["state_dict"]["nominal_sigma_log"])  # DynamicsEnsemble.save: buffers live in state_dict
        if not np.isfinite(sigma):
            raise SystemExit(f"{a.ensemble}: nominal_sigma_log not set (run deploy_calibrate_cusum --set_calibration_from)")
    d = load_chain(a.jobs, a.cycle_s)
    res = run(d, sigma, a)
    res["jobs"] = a.jobs
    table = md_table(res)
    print(json.dumps({k: v for k, v in res.items() if k in ("cycles", "robots", "robots_tune", "robots_test", "robots_baseline_not_done_before_first_onset", "chosen")}, indent=1))
    print(table)
    if a.out:
        json.dump(res, open(a.out + ".json", "w"), indent=1, default=float)
        open(a.out + ".md", "w").write(table + "\n")

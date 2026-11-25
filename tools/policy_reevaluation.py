#!/usr/bin/env python3
"""Empirical comparison of sampling rules in public metabolomics panels.

The companion benchmark module provides data loaders, source checksums and the raw allocation rule.
This script writes only aggregate split-level results (no participant IDs/data).
"""
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LOCAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_ROOT))
import frozen_benchmark as frozen  # noqa: E402

SEED = 20261122
N_OUTER = 400
N_MASKS = 200
M = 6
G_VALUES = (8, 12)
DELTA_V = (0.36, 0.98, 2.0, 4.0, 8.0)
Z975 = 1.959963984540054
T975 = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def tcrit(df):
    return T975.get(max(3, min(30, int(df))), 2.042)


def beta_pilot(xp, yp):
    x0 = xp.mean(axis=0)
    y0 = yp.mean(axis=0)
    sx = xp.std(axis=0, ddof=1)
    sy = yp.std(axis=0, ddof=1)
    sx = np.where(sx > 0, sx, 1.0)
    sy = np.where(sy > 0, sy, 1.0)
    r2, rho = frozen.correlation_squares(xp, yp)
    return rho * sy / sx, x0, y0, float(np.mean(r2))


def beta_oracle(xh, yh):
    xc = xh - xh.mean(axis=0)
    yc = yh - yh.mean(axis=0)
    xx = np.sum(xc * xc, axis=0)
    xy = np.sum(xc * yc, axis=0)
    return np.divide(xy, xx, out=np.zeros_like(xy), where=xx > 0)


def borrow_components(xh, yh, beta, scale):
    """Return equal-feature standardized finite-pool P/E variance components."""
    h = len(yh)
    p = xh * beta[None, :]
    e = yh - p
    pc, ec = p - p.mean(axis=0), e - e.mean(axis=0)
    vp = np.sum(pc * pc, axis=0) / max(h - 1, 1)
    ve = np.sum(ec * ec, axis=0) / max(h - 1, 1)
    cpe = np.sum(pc * ec, axis=0) / max(h - 1, 1)
    scale2 = scale * scale
    return (float(np.mean(vp / scale2)), float(np.mean(ve / scale2)),
            float(np.mean(cpe / scale2)))


def borrow_mse_from_components(components, h, n, k):
    """Exact SRS-without-replacement risk, including the P/E covariance term."""
    vp, ve, cpe = components
    a = 1.0 / n - 1.0 / h
    b = 1.0 / k - 1.0 / h
    return max(0.0, vp * a + ve * b + 2.0 * cpe * a)


def exact_borrow_mse(xh, yh, beta, scale, n, k):
    """Exact finite-pool variance of mean(P over n) + mean(Y-P over nested k)."""
    return borrow_mse_from_components(borrow_components(xh, yh, beta, scale), len(yh), n, k)


def exact_gold_mse(yh, scale, n):
    h = len(yh)
    vy = np.var(yh, axis=0, ddof=1)
    return float(np.mean(np.maximum(vy, 0.0) * (1.0 / n - 1.0 / h) / (scale * scale)))


def feasible_allocations(h, dv, g):
    total = M * (3.0 + dv) + g * (2.0 + dv)
    post = total - M * (3.0 + dv)
    ng = min(h, int(math.floor(post / (2.0 + dv) + 1e-9)))
    if ng < 1:
        raise RuntimeError(f"No feasible matched-gold size: H={h}, dv={dv}, g={g}")
    candidates = []
    for n in range(4, h + 1):
        for k in range(4, n + 1):
            if 2.0 * n + (1.0 + dv) * k <= post + 1e-10:
                candidates.append((n, k))
    return total, post, ng, candidates


def reoptimized_allocation(r, h, dv, total_budget, m):
    """Minimize predicted equal-feature variance under a fixed total budget."""
    post = total_budget - m * (3.0 + dv)
    ng = min(h, int(math.floor(post / (2.0 + dv) + 1e-9)))
    if ng < 1:
        return {"decision": "infeasible", "n_gold": 0, "n_borrow": 0, "k_borrow": 0,
                "post_pilot_budget": post}
    candidates = []
    for n in range(4, h + 1):
        for k in range(4, n + 1):
            if 2.0 * n + (1.0 + dv) * k <= post + 1e-10:
                candidates.append(((1.0 - r) / k + r / n, n, k))
    if not candidates:
        return {"decision": "gold", "n_gold": ng, "n_borrow": 0, "k_borrow": 0,
                "post_pilot_budget": post}
    _, nb, kb = min(candidates, key=lambda q: (q[0], q[1], q[2]))
    decision = "borrow" if ((1.0 - r) / kb + r / nb) < (1.0 / ng - 1e-12) else "gold"
    return {"decision": decision, "n_gold": ng, "n_borrow": nb, "k_borrow": kb,
            "post_pilot_budget": post}


def summarize_outer(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        item = dict(zip(keys, key))
        item["outer_splits"] = len(group)
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in group], dtype=float)
            item[metric + "_mean"] = float(np.mean(vals))
            item[metric + "_split_p025"] = float(np.quantile(vals, .025))
            item[metric + "_split_p50"] = float(np.quantile(vals, .5))
            item[metric + "_split_p975"] = float(np.quantile(vals, .975))
        out.append(item)
    return out


def interval_metrics(ytrue, estimate, se2, df, scale):
    se = np.sqrt(np.maximum(se2, 0.0))
    half_z = Z975 * se
    half_t = tcrit(df) * se
    cov_z = (ytrue >= estimate - half_z) & (ytrue <= estimate + half_z)
    cov_t = (ytrue >= estimate - half_t) & (ytrue <= estimate + half_t)
    return (float(np.mean(cov_z)), float(np.mean(2 * half_z / scale)),
            float(np.mean(cov_t)), float(np.mean(2 * half_t / scale)))


def run_split(data, pilot_idx, held_idx, dv, g, rng):
    x, y = data["X"], data["Y"]
    xp, yp, xh, yh = x[pilot_idx], y[pilot_idx], x[held_idx], y[held_idx]
    h, f = yh.shape
    beta_p, mean_xp, mean_yp, rbar = beta_pilot(xp, yp)
    beta_o = beta_oracle(xh, yh)
    scale = np.std(y, axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    total, post, ng, candidates = feasible_allocations(h, dv, g)
    palloc = frozen.predicted_allocation(rbar, h, dv, g)
    if palloc["decision"] == "borrow":
        p_kind, pn, pk = "borrow", palloc["n_borrow"], palloc["k_borrow"]
    else:
        p_kind, pn, pk = "gold", ng, ng
    pilot_r2, _ = frozen.correlation_squares(xp, yp)
    # One-predictor adjusted R² (n=6), clipped per feature to [0,1].
    r_adj = float(np.mean(np.clip(1.0 - (1.0 - pilot_r2) * (M - 1.0) / (M - 2.0), 0.0, 1.0)))
    # Slope shrinkage lambda=(m-2)/(m-1); the implied standardized predictive
    # variance-reduction fraction is (2 lambda - lambda^2) times pilot R².
    shrink_lambda = (M - 2.0) / (M - 1.0)
    r_shrink = float(np.mean(pilot_r2 * (2.0 * shrink_lambda - shrink_lambda ** 2)))
    adj_alloc = reoptimized_allocation(r_adj, h, dv, total, M)
    shrink_alloc = reoptimized_allocation(r_shrink, h, dv, total, M)

    def alloc_triplet(allocation, slope):
        if allocation["decision"] == "borrow":
            return ("borrow", allocation["n_borrow"], allocation["k_borrow"], slope, allocation)
        return ("gold", allocation["n_gold"], allocation["n_gold"], slope, allocation)

    policies = {
        "baseline": (p_kind, pn, pk, beta_p, palloc),
        "adjusted_r2": alloc_triplet(adj_alloc, beta_p),
        "slope_shrink": alloc_triplet(shrink_alloc, beta_p * shrink_lambda),
        "adjusted_r2_shrink": alloc_triplet(adj_alloc, beta_p * shrink_lambda),
    }

    gold_oracle = exact_gold_mse(yh, scale, ng)
    oracle_components = borrow_components(xh, yh, beta_o, scale)
    oracle_candidates = [(gold_oracle, "gold", ng, ng)]
    for n, k in candidates:
        oracle_candidates.append((borrow_mse_from_components(oracle_components, h, n, k), "borrow", n, k))
    oracle_risk, o_kind, on, ok = min(oracle_candidates, key=lambda q: (q[0], q[1], q[2], q[3]))
    raw_scale = np.ones_like(scale)
    raw_gold = exact_gold_mse(yh, raw_scale, ng)
    raw_oracle_components = borrow_components(xh, yh, beta_o, raw_scale)
    raw_oracle_candidates = [(raw_gold, "gold", ng, ng)]
    for n, k in candidates:
        raw_oracle_candidates.append((borrow_mse_from_components(raw_oracle_components, h, n, k), "borrow", n, k))
    raw_oracle_risk, raw_o_kind, raw_on, raw_ok = min(raw_oracle_candidates, key=lambda q: (q[0], q[1], q[2], q[3]))

    def risk(kind, n, k, beta, score_scale=scale):
        if kind == "gold":
            return exact_gold_mse(yh, score_scale, n)
        return exact_borrow_mse(xh, yh, beta, score_scale, n, k)

    crossed = {
        "pilot_alloc_pilot_slope_mse": risk(p_kind, pn, pk, beta_p),
        "pilot_alloc_oracle_slope_mse": risk(p_kind, pn, pk, beta_o),
        "oracle_alloc_pilot_slope_mse": risk(o_kind, on, ok, beta_p),
        "oracle_alloc_oracle_slope_mse": risk(o_kind, on, ok, beta_o),
        "matched_gold_mse": gold_oracle,
    }

    # Use common nested masks drawn without replacement.
    # Repeated masks quantify design randomization only, not cohort uncertainty.
    perm = np.argsort(rng.random((N_MASKS, h)), axis=1)
    p = (xh - mean_xp[None, :]) * beta_p[None, :] + mean_yp[None, :]
    e = yh - p
    gold_idx = perm[:, :ng]
    gold_est = np.mean(yh[gold_idx], axis=1)
    borrow_idx = perm[:, :pn] if p_kind == "borrow" else perm[:, :max([n for n, _ in candidates], default=ng)]
    # The palloc borrowing arm is needed even when the rule chose gold, to keep
    # diagnostics for the pilot-optimized allocation separate from policy loss.
    if palloc["n_borrow"] > 0:
        bn = palloc["n_borrow"]
        bk = palloc["k_borrow"]
        bidx_n, bidx_k = perm[:, :bn], perm[:, :bk]
        forced_est = np.mean(p[bidx_n], axis=1) + np.mean(e[bidx_k], axis=1)
    else:
        bn = bk = 0
        forced_est = np.full_like(gold_est, np.nan)
    if p_kind == "borrow":
        policy_est = np.mean(p[perm[:, :pn]], axis=1) + np.mean(e[perm[:, :pk]], axis=1)
        ip_n, ip_k = perm[:, :pn], perm[:, :pk]
        pobs, eobs = p[ip_n], e[ip_k]
        vp = np.var(pobs, axis=1, ddof=1)
        ve = np.var(eobs, axis=1, ddof=1)
        cpe = np.mean((pobs[:, :pk] - np.mean(pobs[:, :pk], axis=1, keepdims=True)) *
                      (eobs - np.mean(eobs, axis=1, keepdims=True)), axis=1) * (pk / max(pk - 1, 1))
        fac_n, fac_k = 1.0 / pn - 1.0 / h, 1.0 / pk - 1.0 / h
        se2 = np.maximum(vp * fac_n + ve * fac_k + 2 * cpe * fac_n, 0.0)
        df = min(pn - 1, pk - 1)
    else:
        policy_est = gold_est
        yobs = yh[gold_idx]
        vy = np.var(yobs, axis=1, ddof=1)
        se2 = np.maximum(vy * (1.0 / ng - 1.0 / h), 0.0)
        df = ng - 1

    def mask_estimates(kind, n, k, beta):
        if kind != "borrow":
            return np.mean(yh[perm[:, :n]], axis=1)
        pred = (xh - mean_xp[None, :]) * beta[None, :] + mean_yp[None, :]
        resid = yh - pred
        return np.mean(pred[perm[:, :n]], axis=1) + np.mean(resid[perm[:, :k]], axis=1)

    policy_estimates = {name: mask_estimates(kind, n, k, beta)
                        for name, (kind, n, k, beta, _) in policies.items()}
    policy_est = policy_estimates["baseline"]

    truth_h = np.mean(yh, axis=0)
    scale2 = scale[None, :] ** 2
    pol_mask_mse = np.mean(((policy_est - truth_h[None, :]) / scale[None, :]) ** 2, axis=1)
    gold_mask_mse = np.mean(((gold_est - truth_h[None, :]) / scale[None, :]) ** 2, axis=1)
    forced_mask_mse = np.mean(((forced_est - truth_h[None, :]) / scale[None, :]) ** 2, axis=1) if bn else np.full(N_MASKS, np.nan)
    policy_cov_z, policy_width_z, policy_cov_t, policy_width_t = interval_metrics(
        truth_h[None, :], policy_est, se2, df, scale[None, :])

    # Same-full-roster target. Include measured pilot target outcomes at their
    # pre-specified m/(m+H) weight, then combine the held-out estimate.
    n_roster = len(y)
    pilot_full_mean = np.mean(yp, axis=0)
    policy_roster = (M / n_roster) * pilot_full_mean[None, :] + (h / n_roster) * policy_est
    direct_n = min(n_roster, max(1, int(math.floor(total / (2.0 + dv) + 1e-9))))
    roster_perm = np.argsort(rng.random((N_MASKS, n_roster)), axis=1)
    direct_est = np.mean(y[roster_perm[:, :direct_n]], axis=1)
    truth_roster = np.mean(y, axis=0)
    roster_policy_mse = np.mean(((policy_roster - truth_roster[None, :]) / scale[None, :]) ** 2, axis=1)
    roster_direct_mse = np.mean(((direct_est - truth_roster[None, :]) / scale[None, :]) ** 2, axis=1)

    policy_mse = {}
    roster_mse = {}
    for name, (kind, n, k, beta, _) in policies.items():
        estimates = policy_estimates[name]
        mask_mse = np.mean(((estimates - truth_h[None, :]) / scale[None, :]) ** 2, axis=1)
        exact = risk(kind, n, k, beta)
        roster_est = (M / n_roster) * pilot_full_mean[None, :] + (h / n_roster) * estimates
        roster_score = np.mean(((roster_est - truth_roster[None, :]) / scale[None, :]) ** 2, axis=1)
        policy_mse[name] = (kind, n, k, exact, mask_mse)
        roster_mse[name] = roster_score

    return {
        "n_eval": h, "n_features": f, "pilot_mean_r2": rbar,
        "pilot_adjusted_r2_mean_clipped": r_adj, "pilot_shrink_effective_r2": r_shrink,
        "policy_decision": p_kind, "pilot_n": pn, "pilot_k": pk,
        "adjusted_r2_decision": policies["adjusted_r2"][0],
        "adjusted_r2_n": policies["adjusted_r2"][1], "adjusted_r2_k": policies["adjusted_r2"][2],
        "slope_shrink_decision": policies["slope_shrink"][0],
        "slope_shrink_n": policies["slope_shrink"][1], "slope_shrink_k": policies["slope_shrink"][2],
        "adjusted_r2_shrink_decision": policies["adjusted_r2_shrink"][0],
        "adjusted_r2_shrink_n": policies["adjusted_r2_shrink"][1], "adjusted_r2_shrink_k": policies["adjusted_r2_shrink"][2],
        "oracle_decision": o_kind, "oracle_n": on, "oracle_k": ok,
        "total_budget": total, "post_pilot_budget": post,
        **crossed,
        "oracle_regret_exact_mse": crossed["pilot_alloc_pilot_slope_mse"] - oracle_risk,
        "pilot_alloc_oracle_slope_regret": crossed["pilot_alloc_oracle_slope_mse"] - oracle_risk,
        "oracle_allocation_pilot_slope_regret": crossed["oracle_alloc_pilot_slope_mse"] - oracle_risk,
        "oracle_min_exact_mse": oracle_risk,
        "raw_unit_oracle_min_mse": raw_oracle_risk,
        "raw_unit_oracle_decision": raw_o_kind,
        "raw_unit_oracle_n": raw_on, "raw_unit_oracle_k": raw_ok,
        "policy_mask_mse_mean": float(np.mean(pol_mask_mse)),
        "policy_mask_mse_p10": float(np.quantile(pol_mask_mse, .1)),
        "policy_mask_mse_p50": float(np.quantile(pol_mask_mse, .5)),
        "policy_mask_mse_p90": float(np.quantile(pol_mask_mse, .9)),
        "gold_mask_mse_mean": float(np.mean(gold_mask_mse)),
        "paired_policy_minus_gold_mse_mean": float(np.mean(pol_mask_mse - gold_mask_mse)),
        "paired_policy_minus_gold_mse_p10": float(np.quantile(pol_mask_mse - gold_mask_mse, .1)),
        "paired_policy_minus_gold_mse_p50": float(np.quantile(pol_mask_mse - gold_mask_mse, .5)),
        "paired_policy_minus_gold_mse_p90": float(np.quantile(pol_mask_mse - gold_mask_mse, .9)),
        "forced_borrow_mask_mse_mean": float(np.nanmean(forced_mask_mse)) if bn else float("nan"),
        "policy_interval_coverage_normal": policy_cov_z,
        "policy_interval_width_normal_std": policy_width_z,
        "policy_interval_coverage_t": policy_cov_t,
        "policy_interval_width_t_std": policy_width_t,
        "same_roster_policy_mse_mean": float(np.mean(roster_policy_mse)),
        "same_roster_direct_mse_mean": float(np.mean(roster_direct_mse)),
        "same_roster_policy_minus_direct_mse_mean": float(np.mean(roster_policy_mse - roster_direct_mse)),
        "same_roster_policy_minus_direct_mse_p10": float(np.quantile(roster_policy_mse - roster_direct_mse, .1)),
        "same_roster_policy_minus_direct_mse_p50": float(np.quantile(roster_policy_mse - roster_direct_mse, .5)),
        "same_roster_policy_minus_direct_mse_p90": float(np.quantile(roster_policy_mse - roster_direct_mse, .9)),
        "same_roster_direct_n": direct_n,
        "pilot_fraction_of_roster": M / n_roster,
        "heldout_fraction_of_roster": h / n_roster,
        "baseline_exact_policy_mse": policy_mse["baseline"][3],
        "adjusted_r2_exact_policy_mse": policy_mse["adjusted_r2"][3],
        "slope_shrink_exact_policy_mse": policy_mse["slope_shrink"][3],
        "adjusted_r2_shrink_exact_policy_mse": policy_mse["adjusted_r2_shrink"][3],
        "baseline_raw_unit_exact_mse": risk(policies["baseline"][0], policies["baseline"][1], policies["baseline"][2], policies["baseline"][3], raw_scale),
        "adjusted_r2_raw_unit_exact_mse": risk(policies["adjusted_r2"][0], policies["adjusted_r2"][1], policies["adjusted_r2"][2], policies["adjusted_r2"][3], raw_scale),
        "slope_shrink_raw_unit_exact_mse": risk(policies["slope_shrink"][0], policies["slope_shrink"][1], policies["slope_shrink"][2], policies["slope_shrink"][3], raw_scale),
        "adjusted_r2_shrink_raw_unit_exact_mse": risk(policies["adjusted_r2_shrink"][0], policies["adjusted_r2_shrink"][1], policies["adjusted_r2_shrink"][2], policies["adjusted_r2_shrink"][3], raw_scale),
        "adjusted_r2_minus_baseline_exact_mse": policy_mse["adjusted_r2"][3] - policy_mse["baseline"][3],
        "slope_shrink_minus_baseline_exact_mse": policy_mse["slope_shrink"][3] - policy_mse["baseline"][3],
        "adjusted_r2_shrink_minus_baseline_exact_mse": policy_mse["adjusted_r2_shrink"][3] - policy_mse["baseline"][3],
        "adjusted_r2_mask_mse_mean": float(np.mean(policy_mse["adjusted_r2"][4])),
        "slope_shrink_mask_mse_mean": float(np.mean(policy_mse["slope_shrink"][4])),
        "paired_adjusted_r2_minus_baseline_mask_mse_mean": float(np.mean(policy_mse["adjusted_r2"][4] - policy_mse["baseline"][4])),
        "paired_slope_shrink_minus_baseline_mask_mse_mean": float(np.mean(policy_mse["slope_shrink"][4] - policy_mse["baseline"][4])),
        "adjusted_r2_shrink_mask_mse_mean": float(np.mean(policy_mse["adjusted_r2_shrink"][4])),
        "paired_adjusted_r2_shrink_minus_baseline_mask_mse_mean": float(np.mean(policy_mse["adjusted_r2_shrink"][4] - policy_mse["baseline"][4])),
        "same_roster_adjusted_r2_policy_mse_mean": float(np.mean(roster_mse["adjusted_r2"])),
        "same_roster_slope_shrink_policy_mse_mean": float(np.mean(roster_mse["slope_shrink"])),
        "same_roster_adjusted_r2_shrink_policy_mse_mean": float(np.mean(roster_mse["adjusted_r2_shrink"])),
        "same_roster_adjusted_r2_minus_direct_mse_mean": float(np.mean(roster_mse["adjusted_r2"] - roster_direct_mse)),
        "same_roster_slope_shrink_minus_direct_mse_mean": float(np.mean(roster_mse["slope_shrink"] - roster_direct_mse)),
        "same_roster_adjusted_r2_shrink_minus_direct_mse_mean": float(np.mean(roster_mse["adjusted_r2_shrink"] - roster_direct_mse)),
    }


def lopo_policy(data, dv, g, n_splits=20):
    """Small jackknife sensitivity for the policy-minus-gold exact risk."""
    x0, y0 = data["X"], data["Y"]
    scale0 = np.std(y0, axis=0, ddof=1)
    scale0 = np.where(scale0 > 0, scale0, 1.0)
    n = len(y0)
    outputs = []
    for left_out in range(n):
        keep = np.array([i for i in range(n) if i != left_out], dtype=int)
        x, y = x0[keep], y0[keep]
        rng = np.random.default_rng(SEED + 88_000_000 + left_out * 100 + int(dv * 100) + g)
        deltas = []
        for _ in range(n_splits):
            perm = rng.permutation(len(y))
            pilot, held = perm[:M], perm[M:]
            beta, _, _, rbar = beta_pilot(x[pilot], y[pilot])
            total, post, ng, _ = feasible_allocations(len(held), dv, g)
            alloc = frozen.predicted_allocation(rbar, len(held), dv, g)
            if alloc["decision"] == "borrow":
                risk = exact_borrow_mse(x[held], y[held], beta, scale0, alloc["n_borrow"], alloc["k_borrow"])
            else:
                risk = exact_gold_mse(y[held], scale0, ng)
            gold = exact_gold_mse(y[held], scale0, ng)
            deltas.append(risk - gold)
        outputs.append(float(np.mean(deltas)))
    mean = float(np.mean(outputs))
    se = math.sqrt((n - 1) / n * float(np.sum((np.asarray(outputs) - mean) ** 2)))
    return {"dataset": data["dataset"], "delta_v": dv, "g": g, "left_out_participants": n,
            "splits_per_jackknife_sample": n_splits, "mean_policy_minus_gold_exact_mse": mean,
            "jackknife_se": se, "jackknife_p025": float(np.quantile(outputs, .025)),
            "jackknife_p50": float(np.quantile(outputs, .5)), "jackknife_p975": float(np.quantile(outputs, .975)),
            "jackknife_min": min(outputs), "jackknife_max": max(outputs)}


def m_sensitivity(data, n_splits=80):
    """Optimise the designs for m=4, 6, or 8 at the same total budget."""
    x, y = data["X"], data["Y"]
    scale = np.std(y, axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    n_roster = len(y)
    rows = []
    for dv in DELTA_V:
        for g in G_VALUES:
            fixed_b = M * (3.0 + dv) + g * (2.0 + dv)
            for m in (4, 6, 8):
                rng = np.random.default_rng(SEED + 77_000_000 + int(dv * 1000) + g * 10 + m + n_roster)
                for split_no in range(1, n_splits + 1):
                    perm = rng.permutation(n_roster)
                    pilot, held = perm[:m], perm[m:]
                    xp, yp, xh, yh = x[pilot], y[pilot], x[held], y[held]
                    beta, _, _, rbar = beta_pilot(xp, yp)
                    r2, _ = frozen.correlation_squares(xp, yp)
                    r_adj = float(np.mean(np.clip(1.0 - (1.0 - r2) * (m - 1.0) / (m - 2.0), 0.0, 1.0)))
                    lam = (m - 2.0) / (m - 1.0)
                    r_shrink = float(np.mean(r2 * (2.0 * lam - lam * lam)))
                    for strategy, r_input, beta_use in (
                        ("baseline_r2", rbar, beta),
                        ("adjusted_r2", r_adj, beta),
                        ("slope_shrink", r_shrink, beta * lam),
                        ("adjusted_r2_shrink", r_adj, beta * lam),
                    ):
                        alloc = reoptimized_allocation(r_input, len(held), dv, fixed_b, m)
                        if alloc["decision"] == "infeasible":
                            rows.append({"dataset": data["dataset"], "delta_v": dv, "g": g, "m": m,
                                         "outer_split": split_no, "strategy": strategy, "feasible": 0,
                                         "policy_minus_gold_exact_mse": float("nan"), "decision": "infeasible"})
                            continue
                        if alloc["decision"] == "borrow":
                            pol_risk = exact_borrow_mse(xh, yh, beta_use, scale, alloc["n_borrow"], alloc["k_borrow"])
                        else:
                            pol_risk = exact_gold_mse(yh, scale, alloc["n_gold"])
                        gold_risk = exact_gold_mse(yh, scale, alloc["n_gold"])
                        rows.append({"dataset": data["dataset"], "delta_v": dv, "g": g, "m": m,
                                     "outer_split": split_no, "strategy": strategy, "feasible": 1,
                                     "decision": alloc["decision"], "n_eval": len(held),
                                     "n_gold": alloc["n_gold"], "n_borrow": alloc["n_borrow"],
                                     "k_borrow": alloc["k_borrow"], "fixed_total_budget": fixed_b,
                                     "policy_exact_mse": pol_risk, "gold_exact_mse": gold_risk,
                                     "policy_minus_gold_exact_mse": pol_risk - gold_risk})
    return rows


def main(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("VERIFYING_FROZEN_SOURCE_HASHES", flush=True)
    hashes = frozen.verify_sources()
    print("SOURCE_HASHES_OK loading PoMet primary", flush=True)
    pomet = frozen.pomet_source("pomet_205866.csv", "PoMet_205866_primary")
    print("LOADING PoMet sensitivity", flush=True)
    pomet_sens = frozen.pomet_source("pomet_196764.csv", "PoMet_196764_sensitivity", common_subjects=pomet["subjects"])
    print("LOADING ST003348 JSON and factors", flush=True)
    st = frozen.st_source("ST003348_quant.json", "st003348_factors.json", "ST003348_REST_STAT_REC3_secondary", ("rest", "stat", "rec3"))
    st_new = frozen.st_source("ST003348_quant.json", "st003348_factors.json", "ST003348_REST_REC3_REC22_exploratory_new_visit", ("rest", "rec3", "rec22"))
    print("ALL_DATASETS_LOADED", flush=True)
    datasets = [pomet, pomet_sens, st, st_new]
    split_rng = np.random.default_rng(SEED)
    # Keep the completed run's block-wise schedule so the released source
    # reproduces its outputs: all PoMet permutations, followed by all ST.
    pomet_splits = [split_rng.permutation(len(pomet["subjects"])) for _ in range(N_OUTER)]
    st_splits = [split_rng.permutation(len(st["subjects"])) for _ in range(N_OUTER)]
    splits = {"pomet": pomet_splits, "st": st_splits}

    flow = []
    all_avail = []
    structures = {}
    for data in datasets:
        statuses = Counter(r["status"] for r in data["availability"])
        flow.append({"dataset": data["dataset"], "source_roster_subjects": len(data["subjects"]),
                     "included_features": len(data["features"]),
                     "candidate_features": sum(r["status"] != "excluded by protocol" for r in data["availability"]),
                     "excluded_by_protocol": sum(r["status"] == "excluded by protocol" for r in data["availability"]),
                     "missing_nonfinite_excluded": sum(v for k, v in statuses.items() if "missing/non-finite" in k),
                     "nonpositive_excluded": sum(v for k, v in statuses.items() if "non-positive" in k),
                     "fully_unmeasured_subjects_excluded": data["structure"].get("excluded_fully_unmeasured_triplets", 0),
                     "panel_rule": "common complete positive values at all required visits across the source roster"})
        all_avail.extend(data["availability"])
        structures[data["dataset"]] = data["structure"]

    split_rows = []
    for di, data in enumerate(datasets, start=1):
        family = "pomet" if data["dataset"].startswith("PoMet") else "st"
        print(f"START dataset={data['dataset']} subjects={len(data['subjects'])} features={len(data['features'])}", flush=True)
        for split_no, perm in enumerate(splits[family], start=1):
            pilot_idx, held_idx = perm[:M], perm[M:]
            for dv in DELTA_V:
                for g in G_VALUES:
                    rng = np.random.default_rng(SEED + di * 1_000_000 + split_no * 1000 + int(dv * 100) * 10 + g)
                    result = run_split(data, pilot_idx, held_idx, dv, g, rng)
                    split_rows.append({"dataset": data["dataset"], "outer_split": split_no,
                                       "delta_v": dv, "g": g, **result})
            if split_no % 100 == 0:
                print(f"PROGRESS dataset={data['dataset']} splits={split_no}/{N_OUTER} elapsed_s={time.time()-started:.1f}", flush=True)
    write_csv(outdir / "split_policy_cross.csv", split_rows)
    write_csv(outdir / "availability_flow.csv", flow)
    write_csv(outdir / "feature_availability.csv", all_avail)
    metrics = [k for k in split_rows[0] if k not in {"dataset", "outer_split", "delta_v", "g", "n_eval", "n_features",
                                                       "policy_decision", "oracle_decision", "adjusted_r2_decision",
                                                       "slope_shrink_decision", "adjusted_r2_shrink_decision",
                                                       "raw_unit_oracle_decision"}]
    summary = summarize_outer(split_rows, ["dataset", "delta_v", "g"], metrics)
    # Decision/frequency and rank summaries that are categorical.
    by = defaultdict(list)
    for row in split_rows:
        by[(row["dataset"], row["delta_v"], row["g"])].append(row)
    for srow in summary:
        group = by[(srow["dataset"], srow["delta_v"], srow["g"])]
        srow["pilot_borrow_fraction"] = float(np.mean([r["policy_decision"] == "borrow" for r in group]))
        srow["adjusted_r2_borrow_fraction"] = float(np.mean([r["adjusted_r2_decision"] == "borrow" for r in group]))
        srow["slope_shrink_borrow_fraction"] = float(np.mean([r["slope_shrink_decision"] == "borrow" for r in group]))
        srow["adjusted_r2_shrink_borrow_fraction"] = float(np.mean([r["adjusted_r2_shrink_decision"] == "borrow" for r in group]))
        srow["raw_unit_oracle_borrow_fraction"] = float(np.mean([r["raw_unit_oracle_decision"] == "borrow" for r in group]))
        srow["oracle_borrow_fraction"] = float(np.mean([r["oracle_decision"] == "borrow" for r in group]))
        srow["oracle_allocation_matches_pilot_fraction"] = float(np.mean([
            r["policy_decision"] == r["oracle_decision"] and r["pilot_n"] == r["oracle_n"] and r["pilot_k"] == r["oracle_k"] for r in group]))
    write_csv(outdir / "summary_policy_cross.csv", summary)

    lopo = []
    primary = [pomet, st]
    for data in primary:
        for dv in (0.98, 2.0, 8.0):
            for g in G_VALUES:
                print(f"LOPO dataset={data['dataset']} dv={dv} g={g}", flush=True)
                lopo.append(lopo_policy(data, dv, g, n_splits=20))
    write_csv(outdir / "lopo_policy_sensitivity.csv", lopo)

    m_rows = []
    for data in primary:
        print(f"M_SENSITIVITY dataset={data['dataset']} fixed_B_from_m6 n_splits=80", flush=True)
        m_rows.extend(m_sensitivity(data, n_splits=80))
    write_csv(outdir / "m_sensitivity_split.csv", m_rows)
    m_valid = [r for r in m_rows if r["feasible"] == 1]
    m_summary = summarize_outer(m_valid, ["dataset", "delta_v", "g", "m", "strategy"],
                                ["policy_exact_mse", "gold_exact_mse", "policy_minus_gold_exact_mse"])
    for srow in m_summary:
        group = [r for r in m_valid if (r["dataset"], r["delta_v"], r["g"], r["m"], r["strategy"]) ==
                 (srow["dataset"], srow["delta_v"], srow["g"], srow["m"], srow["strategy"])]
        srow["borrow_fraction"] = float(np.mean([r["decision"] == "borrow" for r in group]))
    write_csv(outdir / "m_sensitivity_summary.csv", m_summary)

    protocol = {
        "source_hashes_sha256": hashes, "seed": SEED, "outer_splits": N_OUTER,
        "nested_random_masks_per_split": N_MASKS, "pilot_m": M, "g_values": list(G_VALUES),
        "delta_v_grid": list(DELTA_V), "split_protocol": "400 partitions per data source with seed 20261122, generating all PoMet partitions before the ST003348 partitions.",
        "oracle": "diagnostic only; held-out finite-pool slope and exact finite-population risk are used to rank every feasible integer N,k plus matched-gold",
        "exact_finite_pool_variance": "Var(mean_N P + mean_k E) = S_P^2(1/N-1/H)+S_E^2(1/k-1/H)+2 S_PE(1/N-1/H); equal-feature scaling uses source target SD",
        "crossed_comparison": "pilot versus held-out oracle slope crossed with pilot policy allocation versus exact finite-pool oracle allocation",
        "post_hoc_policy_diagnostics": "Adjusted R2 is 1-(1-r2)(m-1)/(m-2), clipped feature-wise to [0,1] then averaged. Slope shrinkage uses lambda=(m-2)/(m-1), with allocation optimized using (2lambda-lambda^2) times pilot r2. Adjusted-only, shrink-only, and combined adjusted-R2 plus shrunken-slope policies each re-optimize allocation under the same B; they are exploratory and checked on these same datasets.",
        "paid_pilot_size_sensitivity": "m=4/6/8 reoptimized under the total budget for the m=6 design at each g and delta_v; 80 reproducible splits per primary source and condition.",
        "same_roster_target": "pilot target mean weighted m/(m+H) plus held-out policy estimate weighted H/(m+H), compared with no-pilot random full-roster gold sample under full B",
        "intervals": "exploratory finite-pool plug-in intervals conditional on selected policy and pilot slope; normal 1.96 and t with df=min(n-1,k-1) for borrow, n-1 for gold",
        "uncertainty": "mask percentiles characterize randomization conditional on split; split percentiles characterize repeated random partitions of the same small cohort, not population CIs; leave-one-participant-out sensitivity uses 20 splits per deletion for PoMet primary and ST main at selected costs",
        "target_availability": "The retrospective panel contains features with complete positive values at all required visits. Counts and reasons for exclusion are reported. This panel defines the analytical sample.",
        "individual_data_exported": False,
        "structures": structures,
    }
    (outdir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    meta = {"elapsed_seconds": time.time()-started, "split_rows": len(split_rows), "lopo_rows": len(lopo),
            "datasets": {d["dataset"]: {"subjects": len(d["subjects"]), "features": len(d["features"])} for d in datasets},
        "result_files": ["split_policy_cross.csv", "summary_policy_cross.csv", "availability_flow.csv",
                             "feature_availability.csv", "lopo_policy_sensitivity.csv", "m_sensitivity_split.csv",
                             "m_sensitivity_summary.csv", "protocol.json"]}
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE elapsed_s={meta['elapsed_seconds']:.1f} split_rows={len(split_rows)} lopo_rows={len(lopo)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

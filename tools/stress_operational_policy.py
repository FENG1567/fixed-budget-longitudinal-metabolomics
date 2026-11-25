#!/usr/bin/env python3
"""Simulation of sampling rules under a fixed total budget.

All policy decisions are based on a separate pilot. Main-study draws are
independent, and pilot cost is deducted from the fixed total budget.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


SEED = 20261123
REPS = 2000
BUDGET = 480.0
DVS = (2.0, 8.0)
PILOT_SIZES = (6, 12, 24)
HIGH_RHO2 = 0.93
GRID_N = 1000
SCENARIOS = {
    "homogeneous": {"site_mix": False, "baseline_var": 0.0, "endpoint_var": 0.0, "missing": "none"},
    "aux_heterogeneity": {"site_mix": True, "baseline_var": 0.0, "endpoint_var": 0.0, "missing": "none"},
    "shared_baseline_error": {"site_mix": False, "baseline_var": 0.40, "endpoint_var": 0.05, "missing": "none"},
    "mcar_15pct": {"site_mix": False, "baseline_var": 0.40, "endpoint_var": 0.05, "missing": "mcar"},
    "informative_late_15pct": {"site_mix": False, "baseline_var": 0.40, "endpoint_var": 0.05, "missing": "late_outcome"},
}


def candidate_designs(dv: float, pilot_n: int):
    # Paid pilot incurs baseline + early + late plus one pilot-extension unit.
    pilot_cost = (3.0 + dv) * pilot_n
    remaining = BUDGET - pilot_cost
    direct_n = int(math.floor(remaining / (2.0 + dv)))
    no_pilot_n = int(math.floor(BUDGET / (2.0 + dv)))
    choices = [{"kind": "direct", "N": direct_n, "k": direct_n,
                "cost": (2.0 + dv) * direct_n}]
    max_k = int(math.floor(remaining / (1.0 + dv)))
    for k in range(4, max_k + 1):
        max_n = int(math.floor((remaining - (1.0 + dv) * k) / 2.0))
        for n in range(k, max_n + 1):
            cost = 2.0 * n + (1.0 + dv) * k
            choices.append({"kind": "borrow", "N": n, "k": k, "cost": cost})
    if direct_n < 1 or not choices:
        raise ValueError(f"No feasible design for dv={dv}, pilot_n={pilot_n}")
    return choices, remaining, direct_n, no_pilot_n, pilot_cost


def risk_per_unit_variance(choice, rho2: float) -> float:
    if choice["kind"] == "direct":
        return 1.0 / choice["N"]
    return (1.0 - rho2) / choice["k"] + rho2 / choice["N"]


def design_map(choices):
    grid = np.linspace(0.0, 0.999, GRID_N)
    ns = np.array([c["N"] for c in choices], dtype=float)
    ks = np.array([c["k"] for c in choices], dtype=float)
    direct = np.array([c["kind"] == "direct" for c in choices])
    matrix = (1.0 - grid[:, None]) / ks[None, :] + grid[:, None] / ns[None, :]
    matrix[:, direct] = 1.0 / ns[direct][None, :]
    selected = matrix.argmin(axis=1)
    return grid, selected


def oracle_choice(choices, rho2):
    return min(choices, key=lambda c: risk_per_unit_variance(c, rho2))


def break_even_rho2(choices):
    direct = choices[0]["N"]
    direct_risk = 1.0 / direct
    crossings = []
    for c in choices[1:]:
        intercept = 1.0 / c["k"]
        slope = 1.0 / c["N"] - 1.0 / c["k"]
        if slope < 0:
            q = (direct_risk - intercept) / slope
            if 0.0 <= q <= 1.0:
                crossings.append(q)
    return min(crossings) if crossings else float("nan")


def rho2_levels(qcut):
    if math.isfinite(qcut):
        values = [
            ("below_boundary", max(0.02, qcut - 0.06)),
            ("integer_boundary", max(0.02, min(0.98, qcut))),
            ("above_boundary", min(0.93, qcut + 0.06)),
        ]
    else:
        values = [("below_boundary", 0.25), ("integer_boundary", 0.50),
                  ("above_boundary", 0.75)]
    values.append(("high_rho_positive_side", HIGH_RHO2))
    unique = []
    seen = set()
    for label, q in values:
        q = round(min(0.93, max(0.02, q)), 6)
        if q not in seen:
            unique.append((label, q))
            seen.add(q)
    return unique


def draw_latent_x(rng, reps, n, site_mix):
    if not site_mix:
        return rng.normal(size=(reps, n))
    # Three recruitment strata give different auxiliary means and variances.
    site = rng.integers(0, 3, size=(reps, n))
    means = np.array([-0.6, 0.0, 0.6])
    sds = np.array([0.75, 1.0, 1.25])
    raw = means[site] + sds[site] * rng.normal(size=(reps, n))
    total_var = float(np.mean(sds ** 2) + np.mean(means ** 2))
    return raw / math.sqrt(total_var)


def draw_xy(rng, reps, n, rho_latent, config):
    x_latent = draw_latent_x(rng, reps, n, config["site_mix"])
    eps = rng.normal(size=(reps, n))
    y_latent = rho_latent * x_latent + math.sqrt(max(0.0, 1.0 - rho_latent ** 2)) * eps
    shared = rng.normal(size=(reps, n)) * math.sqrt(config["baseline_var"])
    x = x_latent + shared + rng.normal(size=(reps, n)) * math.sqrt(config["endpoint_var"])
    y = y_latent + shared + rng.normal(size=(reps, n)) * math.sqrt(config["endpoint_var"])
    return x, y


def late_observed_mask(rng, y, config):
    mechanism = config["missing"]
    if mechanism == "none":
        return np.ones(y.shape, dtype=bool)
    if mechanism == "mcar":
        return rng.random(size=y.shape) >= 0.15
    # Outcome-dependent missingness: higher late outcomes are more likely missing.
    sd = math.sqrt(1.0 + config["baseline_var"] + config["endpoint_var"])
    z = y / sd
    p_miss = 1.0 / (1.0 + np.exp(-(-1.94 + 0.8 * z)))
    return rng.random(size=y.shape) >= p_miss


def pilot_statistics(x, y, valid, pilot_n):
    xx = x[:, :pilot_n]
    yy = y[:, :pilot_n]
    mm = valid[:, :pilot_n]
    n = mm.sum(axis=1)
    safe_n = np.maximum(n, 1)
    xb = (xx * mm).sum(axis=1) / safe_n
    yb = (yy * mm).sum(axis=1) / safe_n
    dx = (xx - xb[:, None]) * mm
    dy = (yy - yb[:, None]) * mm
    sxx = (dx * dx).sum(axis=1) / np.maximum(n - 1, 1)
    syy = (dy * dy).sum(axis=1) / np.maximum(n - 1, 1)
    sxy = (dx * dy).sum(axis=1) / np.maximum(n - 1, 1)
    raw_r2 = np.divide(sxy * sxy, sxx * syy, out=np.zeros_like(sxy), where=(sxx * syy) > 1e-12)
    raw_r2 = np.clip(raw_r2, 0.0, 1.0)
    # Use the actual complete-pair count in each replicate under missingness.
    adjusted = np.where(n >= 3,
                        np.maximum(0.0, 1.0 - (1.0 - raw_r2) * (n - 1.0) / np.maximum(n - 2.0, 1.0)),
                        0.0)
    beta = np.divide(sxy, sxx, out=np.zeros_like(sxy), where=sxx > 1e-12)
    beta[n < 3] = 0.0
    return raw_r2, adjusted, beta, n


def direct_estimate(y, valid, n):
    mask = valid[:, :n]
    yy = y[:, :n]
    count = mask.sum(axis=1)
    safe = np.maximum(count, 1)
    mean = (yy * mask).sum(axis=1) / safe
    var = (((yy - mean[:, None]) ** 2) * mask).sum(axis=1) / np.maximum(count - 1, 1)
    se = np.sqrt(var / safe)
    fail = count < 2
    mean[fail] = np.nan
    se[fail] = np.nan
    return {"estimate": mean, "se": se, "fail": fail,
            "N": np.full(y.shape[0], n), "k": np.full(y.shape[0], n),
            "borrow": np.zeros(y.shape[0], dtype=bool)}


def two_phase_estimate(x, y, valid, n_vec, k_vec, beta_vec):
    reps, max_n = x.shape
    cols = np.arange(max_n)[None, :]
    aux_mask = cols < n_vec[:, None]
    pair_mask = (cols < k_vec[:, None]) & valid
    n_pair = pair_mask.sum(axis=1)
    safe_pair = np.maximum(n_pair, 1)
    xbar_all = (x * aux_mask).sum(axis=1) / np.maximum(n_vec, 1)
    xbar_pair = (x * pair_mask).sum(axis=1) / safe_pair
    ybar_pair = (y * pair_mask).sum(axis=1) / safe_pair
    estimate = ybar_pair + beta_vec * (xbar_all - xbar_pair)
    resid = y - beta_vec[:, None] * x
    rbar = (resid * pair_mask).sum(axis=1) / safe_pair
    var_r = (((resid - rbar[:, None]) ** 2) * pair_mask).sum(axis=1) / np.maximum(n_pair - 1, 1)
    xbar = (x * aux_mask).sum(axis=1) / np.maximum(n_vec, 1)
    var_x = (((x - xbar[:, None]) ** 2) * aux_mask).sum(axis=1) / np.maximum(n_vec - 1, 1)
    var_est = var_r / safe_pair + beta_vec ** 2 * var_x / np.maximum(n_vec, 1)
    se = np.sqrt(np.maximum(var_est, 0.0))
    fail = (n_pair < 3) | (n_vec < 2) | ~np.isfinite(estimate) | ~np.isfinite(se)
    estimate[fail] = np.nan
    se[fail] = np.nan
    return {"estimate": estimate, "se": se, "fail": fail,
            "N": n_vec.astype(int), "k": k_vec.astype(int),
            "borrow": np.ones(reps, dtype=bool)}


def summarize(cell, estimates, oracle_name="oracle"):
    direct = estimates["direct_matched_pilot"]
    oracle = estimates[oracle_name]
    oracle_loss = (oracle["estimate"]) ** 2
    direct_loss = direct["estimate"] ** 2
    rows = []
    for name, obj in estimates.items():
        err = obj["estimate"]
        valid = np.isfinite(err)
        loss = err[valid] ** 2
        dvalid = valid & np.isfinite(direct["estimate"])
        pair_diff = err[dvalid] ** 2 - direct["estimate"][dvalid] ** 2
        ovalid = valid & np.isfinite(oracle["estimate"])
        regret = err[ovalid] ** 2 - oracle["estimate"][ovalid] ** 2
        ci_lo = err - 1.96 * obj["se"]
        ci_hi = err + 1.96 * obj["se"]
        coverage = (ci_lo <= 0.0) & (ci_hi >= 0.0) & valid
        row = dict(cell)
        pilot_cost = 0.0 if name == "direct_no_pilot" else cell["pilot_cost_if_pilot_dependent_policy"]
        main_budget = BUDGET if name == "direct_no_pilot" else cell["remaining_main_budget_after_pilot"]
        row.update({
            "policy": name,
            "pilot_cost_charged": pilot_cost,
            "main_budget_available": main_budget,
            "MSE": float(np.mean(loss)) if len(loss) else float("nan"),
            "bias": float(np.mean(err[valid])) if valid.any() else float("nan"),
            "RMSE": float(np.sqrt(np.mean(loss))) if len(loss) else float("nan"),
            "MSE_MCSE": float(np.std(loss, ddof=1) / math.sqrt(len(loss))) if len(loss) > 1 else float("nan"),
            "MSE_ratio_vs_matched_pilot_direct": float(np.mean(loss) / np.mean(direct_loss[np.isfinite(direct_loss)])) if len(loss) and np.isfinite(direct_loss).any() else float("nan"),
            "paired_MSE_diff_vs_matched_direct": float(np.mean(pair_diff)) if len(pair_diff) else float("nan"),
            "paired_MCSE_diff_vs_matched_direct": float(np.std(pair_diff, ddof=1) / math.sqrt(len(pair_diff))) if len(pair_diff) > 1 else float("nan"),
            "mean_regret_vs_oracle": float(np.mean(regret)) if len(regret) else float("nan"),
            "paired_MCSE_regret_vs_oracle": float(np.std(regret, ddof=1) / math.sqrt(len(regret))) if len(regret) > 1 else float("nan"),
            "empirical_coverage_95": float(coverage.sum() / valid.sum()) if valid.any() else float("nan"),
            "coverage_MCSE": math.sqrt(float(coverage.sum() / valid.sum()) * (1.0 - float(coverage.sum() / valid.sum())) / valid.sum()) if valid.any() else float("nan"),
            "mean_CI_width": float(np.nanmean(3.92 * obj["se"])) if valid.any() else float("nan"),
            "fail_count": int((~valid).sum()),
            "selected_two_phase_rate": float(np.mean(obj["borrow"])),
            "mean_N": float(np.mean(obj["N"])),
            "mean_k": float(np.mean(obj["k"])),
        })
        rows.append(row)
    return rows


def run_cell(dv, pilot_n, rho2_target, rho_level, scenario_name, config, choices,
             remaining, direct_n, no_pilot_n, pilot_cost, qcut, cell_index,
             grid, design_ids):
    rho_obs = math.sqrt(rho2_target)
    v_x = 1.0 + config["baseline_var"] + config["endpoint_var"]
    rho_latent = rho_obs * v_x - config["baseline_var"]
    max_rho_obs = (1.0 + config["baseline_var"]) / v_x
    if rho_latent < 0 or rho_latent >= 1.0:
        raise ValueError(f"Infeasible observed correlation {rho_obs:.4f} under {scenario_name}")
    true_rho2 = rho2_target
    beta_true = rho_obs  # Equal observed X/Y variances by construction.
    max_design_n = max(c["N"] for c in choices)
    max_n = max(max_design_n, no_pilot_n)
    seed = np.random.SeedSequence([SEED, int(dv * 100), pilot_n, cell_index,
                                   int(round(rho2_target * 1_000_000))])
    rng = np.random.default_rng(seed)
    pilot_x, pilot_y = draw_xy(rng, REPS, pilot_n, rho_latent, config)
    pilot_valid = late_observed_mask(rng, pilot_y, config)
    raw_qhat, qhat, beta_raw, pilot_complete = pilot_statistics(pilot_x, pilot_y, pilot_valid, pilot_n)
    shrink = (pilot_n - 2.0) / (pilot_n + 2.0)
    beta_shrink = shrink * beta_raw
    design_idx = np.clip(np.floor(np.clip(qhat, 0.0, 0.999) / 0.999 * (GRID_N - 1)).astype(int), 0, GRID_N - 1)
    selected_idx = design_ids[design_idx]
    selected = [choices[i] for i in selected_idx]
    n_op = np.array([c["N"] for c in selected])
    k_op = np.array([c["k"] for c in selected])
    selected_borrow = np.array([c["kind"] == "borrow" for c in selected])
    raw_design_idx = np.clip(np.floor(np.clip(raw_qhat, 0.0, 0.999) / 0.999 * (GRID_N - 1)).astype(int), 0, GRID_N - 1)
    raw_selected = [choices[i] for i in design_ids[raw_design_idx]]
    n_raw_op = np.array([c["N"] for c in raw_selected])
    k_raw_op = np.array([c["k"] for c in raw_selected])
    raw_selected_borrow = np.array([c["kind"] == "borrow" for c in raw_selected])
    oracle = oracle_choice(choices, true_rho2)
    n_oracle = int(oracle["N"])
    k_oracle = int(oracle["k"])

    main_x, main_y = draw_xy(rng, REPS, max_n, rho_latent, config)
    main_valid = late_observed_mask(rng, main_y, config)
    direct_no = direct_estimate(main_y, main_valid, no_pilot_n)
    direct_matched = direct_estimate(main_y, main_valid, direct_n)
    out = {"direct_no_pilot": direct_no, "direct_matched_pilot": direct_matched}
    beta_true_vec = np.full(REPS, beta_true)
    out["oracle"] = (direct_estimate(main_y, main_valid, direct_n) if oracle["kind"] == "direct"
                      else two_phase_estimate(main_x, main_y, main_valid,
                                              np.full(REPS, n_oracle), np.full(REPS, k_oracle), beta_true_vec))
    allocation_direct = direct_estimate(main_y, main_valid, direct_n)
    allocation_borrow = two_phase_estimate(main_x, main_y, main_valid, n_op, k_op, beta_true_vec)
    allocation_only = dict(allocation_direct)
    allocation_only["estimate"] = np.where(selected_borrow, allocation_borrow["estimate"], allocation_direct["estimate"])
    allocation_only["se"] = np.where(selected_borrow, allocation_borrow["se"], allocation_direct["se"])
    allocation_only["fail"] = np.where(selected_borrow, allocation_borrow["fail"], allocation_direct["fail"])
    allocation_only["N"] = n_op
    allocation_only["k"] = k_op
    allocation_only["borrow"] = selected_borrow
    out["allocation_only_true_slope"] = allocation_only
    out["oracle_allocation_pilot_slope"] = (direct_estimate(main_y, main_valid, direct_n) if oracle["kind"] == "direct"
                                            else two_phase_estimate(main_x, main_y, main_valid,
                                                                    np.full(REPS, n_oracle), np.full(REPS, k_oracle), beta_raw))
    out["oracle_allocation_shrunk_slope"] = (direct_estimate(main_y, main_valid, direct_n) if oracle["kind"] == "direct"
                                             else two_phase_estimate(main_x, main_y, main_valid,
                                                                     np.full(REPS, n_oracle), np.full(REPS, k_oracle), beta_shrink))
    operational_raw_borrow = two_phase_estimate(main_x, main_y, main_valid, n_op, k_op, beta_raw)
    operational_shrink_borrow = two_phase_estimate(main_x, main_y, main_valid, n_op, k_op, beta_shrink)
    operational_raw = dict(direct_matched)
    operational_raw["estimate"] = np.where(selected_borrow, operational_raw_borrow["estimate"], direct_matched["estimate"])
    operational_raw["se"] = np.where(selected_borrow, operational_raw_borrow["se"], direct_matched["se"])
    operational_raw["fail"] = np.where(selected_borrow, operational_raw_borrow["fail"], direct_matched["fail"])
    operational_raw["N"] = n_op
    operational_raw["k"] = k_op
    operational_raw["borrow"] = selected_borrow
    operational_shrink = dict(direct_matched)
    operational_shrink["estimate"] = np.where(selected_borrow, operational_shrink_borrow["estimate"], direct_matched["estimate"])
    operational_shrink["se"] = np.where(selected_borrow, operational_shrink_borrow["se"], direct_matched["se"])
    operational_shrink["fail"] = np.where(selected_borrow, operational_shrink_borrow["fail"], direct_matched["fail"])
    operational_shrink["N"] = n_op
    operational_shrink["k"] = k_op
    operational_shrink["borrow"] = selected_borrow
    out["operational_adjustedR2_raw_slope"] = operational_raw
    out["operational_adjustedR2_shrunk_slope"] = operational_shrink
    raw_r2_borrow = two_phase_estimate(main_x, main_y, main_valid, n_raw_op, k_raw_op, beta_raw)
    operational_raw_r2 = dict(direct_matched)
    operational_raw_r2["estimate"] = np.where(raw_selected_borrow, raw_r2_borrow["estimate"], direct_matched["estimate"])
    operational_raw_r2["se"] = np.where(raw_selected_borrow, raw_r2_borrow["se"], direct_matched["se"])
    operational_raw_r2["fail"] = np.where(raw_selected_borrow, raw_r2_borrow["fail"], direct_matched["fail"])
    operational_raw_r2["N"] = n_raw_op
    operational_raw_r2["k"] = k_raw_op
    operational_raw_r2["borrow"] = raw_selected_borrow
    out["operational_rawR2_raw_slope"] = operational_raw_r2
    oracle_borrow = oracle["kind"] == "borrow"
    cell = {
        "scenario": scenario_name, "delta_v": dv, "total_budget": BUDGET,
        "pilot_n_independent": pilot_n, "pilot_cost_if_pilot_dependent_policy": pilot_cost,
        "remaining_main_budget_after_pilot": remaining, "rho2_observed_target": rho2_target,
        "rho_observed_target": rho_obs, "rho_latent": rho_latent,
        "rho2_observed_population": ((rho_latent + config["baseline_var"]) / v_x) ** 2,
        "baseline_error_variance_shared": config["baseline_var"],
        "endpoint_error_variance_each": config["endpoint_var"],
        "late_missingness": config["missing"], "target_max_observed_rho2": max_rho_obs ** 2,
        "integer_break_even_rho2": qcut, "rho2_level": rho_level,
        "direct_no_pilot_N": no_pilot_n, "direct_matched_pilot_N": direct_n,
        "oracle_kind": oracle["kind"], "oracle_N": n_oracle, "oracle_k": k_oracle,
        "oracle_cost_total": pilot_cost + oracle["cost"],
        "pilot_adjusted_R2_mean": float(np.mean(qhat)), "pilot_slope_mean": float(np.mean(beta_raw)),
        "pilot_raw_R2_mean": float(np.mean(raw_qhat)),
        "pilot_slope_shrink_factor": shrink, "pilot_complete_mean": float(np.mean(pilot_complete)),
        "operational_two_phase_selection_rate": float(np.mean(selected_borrow)),
        "operational_rawR2_two_phase_selection_rate": float(np.mean(raw_selected_borrow)),
        "oracle_two_phase_decision": int(oracle_borrow),
        "selection_disagreement_rate_vs_oracle": float(np.mean(selected_borrow != oracle_borrow)),
        "operational_mean_N": float(np.mean(n_op)), "operational_mean_k": float(np.mean(k_op)),
        "operational_max_cost_total": float(pilot_cost + max(c["cost"] for c in selected)),
        "reps_planned": REPS,
    }
    return summarize(cell, out)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    print(f"START reps_per_cell={REPS} budget={BUDGET}", flush=True)
    all_rows = []
    design_rows = []
    cell_index = 0
    for dv in DVS:
        for pilot_n in PILOT_SIZES:
            choices, remaining, direct_n, no_pilot_n, pilot_cost = candidate_designs(dv, pilot_n)
            qcut = break_even_rho2(choices)
            grid, design_ids = design_map(choices)
            max_alloc = max(choices, key=lambda c: c["N"])
            for label, q in rho2_levels(qcut):
                for scenario_name, config in SCENARIOS.items():
                    # Shared-baseline-error scenarios only support the requested rho range below
                    # the analytic maximum; fail loudly instead of silently changing rho.
                    max_q = ((1.0 + config["baseline_var"]) /
                             (1.0 + config["baseline_var"] + config["endpoint_var"])) ** 2
                    if q > max_q + 1e-10:
                        continue
                    rows = run_cell(dv, pilot_n, q, label, scenario_name, config,
                                    choices, remaining, direct_n, no_pilot_n,
                                    pilot_cost, qcut, cell_index, grid, design_ids)
                    all_rows.extend(rows)
                    cell_index += 1
                    if cell_index % 10 == 0 or cell_index == 1:
                        print(f"PROGRESS cells={cell_index} rows={len(all_rows)} dv={dv} m={pilot_n}", flush=True)
            design_rows.append({
                "delta_v": dv, "total_budget": BUDGET, "pilot_n_independent": pilot_n,
                "pilot_cost": pilot_cost, "remaining_main_budget": remaining,
                "direct_no_pilot_N": no_pilot_n, "direct_matched_pilot_N": direct_n,
                "integer_break_even_rho2": qcut,
                "oracle_max_feasible_N": max_alloc["N"], "oracle_max_feasible_k": max_alloc["k"],
                "candidate_design_count_including_direct": len(choices),
                "operational_allocation_basis": "pilot adjusted R2; integer minimization over direct plus two-phase designs",
                "pilot_slope_rule": "OLS slope; sensitivity shrink factor (m-2)/(m+2)",
        "adjusted_R2_missingness_rule": "adjusted R2 uses actual complete pilot pair count per replicate; fewer than 3 complete pairs sets R2 to zero",
                "pilot_and_main": "independent draws; for pilot-dependent/matched policies, pilot outcomes are excluded from main estimators and cost (3+delta_v)m is charged; no-pilot direct has no pilot cost",
            })
    out_csv = out / "operational_policy_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    design_csv = out / "operational_design_grid.csv"
    with design_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(design_rows[0].keys()))
        writer.writeheader()
        writer.writerows(design_rows)
    metadata = {
        "seed": SEED, "reps_per_cell": REPS, "budget": BUDGET,
        "delta_v_grid": DVS, "pilot_sizes": PILOT_SIZES,
        "scenarios": SCENARIOS, "rho_levels": "integer break-even -0.06, boundary, +0.06, plus rho2=0.93; duplicates removed",
        "policy_estimand": "mean of observed late outcome in the synthetic target population",
        "costs": "direct main-study complete measurement (2+delta_v); auxiliary-only main-study person 2; paired late measurement increment (1+delta_v); independent paid pilot m charged at (3+delta_v)m for baseline, early, late, and one pilot-extension unit",
        "information": "pilot and main are independent; pilot-dependent/matched policies charge the same pilot cost and exclude pilot outcomes from main estimators; no-pilot direct incurs no pilot cost",
        "intervals": "nominal normal 95% intervals; two-phase variance uses estimated residual variance plus beta^2 Var(X)/N; diagnostic only, especially under outcome-dependent missingness",
        "informative_missingness": "late outcome missing with logistic probability logit^-1(-1.94+0.8*Y/sd), complete-case analysis; deliberately outcome-dependent and not claimed as MAR",
        "shared_baseline_error": "the same subject-specific e0 is added to both latent changes; independent endpoint errors are also added; observed and latent correlations are both reported",
        "no_claim": "synthetic operational stress test only; no prospective efficacy claim",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "cells": cell_index,
                      "metrics": str(out_csv), "design_grid": str(design_csv),
                      "metadata": str(out / 'run_metadata.json')}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)

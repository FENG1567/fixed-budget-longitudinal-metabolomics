#!/usr/bin/env python3
"""Reference empirical evaluation of the timepoint design rule."""

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("OMICS_REPLAY_INPUTS", ROOT / "inputs" / "replay"))
OUT = Path(os.environ.get("OMICS_REPLAY_OUTPUT", ROOT / "build" / "interleaved"))
SEED = 20261122
N_OUTER = 400
N_MASKS = 200
PILOT_M = 6
G_VALUES = (8, 12)
DELTA_V = (0.36, 0.98, 2.0, 4.0, 8.0)
EXPECTED_SHA256 = {
    "pomet_205866.csv": "77127e4c3fee223fb0f61caaada2593d0ff4fba48d69949048d9f9347a3dfc08",
    "pomet_196764.csv": "17a4e13f368233322c7bb690fdd92445553ab17e5ce56e6a035f38b9b332eafc",
    "ST003348_quant.json": "f6821b1eecf7f77cee1dd41710b548dc1f3679ced8a1e8ceb271835ec221e55d",
    "st003348_factors.json": "e31bb851873fee1d185a2d91d8b5553637145d799793d416ecf2be6c3f8c3329",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources():
    actual = {}
    for name, expected in EXPECTED_SHA256.items():
        path = SOURCE / name
        got = sha256(path)
        actual[name] = got
        if got != expected:
            raise RuntimeError(f"Source checksum mismatch for {path}: {got} != {expected}")
    return actual


def as_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def pomet_source(filename, label, common_subjects=None):
    path = SOURCE / filename
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        headers = reader.fieldnames or []
        rows = list(reader)
    required_codes = ("1", "7", "8")
    by_subject = defaultdict(dict)
    counts = Counter()
    for row in rows:
        code = str(row.get("time", ""))
        if code in required_codes:
            sid = str(row["id"])
            if code in by_subject[sid]:
                raise RuntimeError(f"Duplicate PoMet row for {sid}, time code {code}")
            by_subject[sid][code] = row
            counts[code] += 1
    excluded_cols = {"id", "sex", "time", "glucose", "insulin"}
    feature_cols = [h for h in headers if h.lower() not in excluded_cols]

    def has_any_measurement(row):
        for feature in feature_cols:
            value = as_float(row.get(feature))
            if value is not None and value > 0:
                return True
        return False

    # A subject with a fully empty required visit is unavailable for paired
    # analysis; partial marker missingness is handled by the fixed feature rule.
    complete = sorted(sid for sid, visits in by_subject.items()
                      if all(code in visits for code in required_codes)
                      and all(has_any_measurement(visits[code]) for code in required_codes))
    if common_subjects is not None:
        missing = sorted(set(common_subjects) - set(complete))
        if missing:
            raise RuntimeError(f"{filename} lacks {len(missing)} expected matched triplets")
        complete = list(common_subjects)
    included, availability, Xcols, Ycols = [], [], [], []
    for metadata_feature in headers:
        if metadata_feature.lower() in {"glucose", "insulin"}:
            availability.append({"dataset": label, "feature": metadata_feature,
                                 "status": "excluded by protocol"})
    for feature in feature_cols:
        vals = {code: [] for code in required_codes}
        reason = "included"
        for sid in complete:
            for code in required_codes:
                value = as_float(by_subject[sid][code].get(feature))
                if value is None:
                    reason = "excluded: missing/non-finite at required visit"
                    break
                if value <= 0:
                    reason = "excluded: non-positive concentration prevents log2 change"
                    break
                vals[code].append(value)
            if reason != "included":
                break
        availability.append({"dataset": label, "feature": feature, "status": reason})
        if reason == "included":
            base = np.asarray(vals["1"], dtype=float)
            aux = np.asarray(vals["7"], dtype=float)
            late = np.asarray(vals["8"], dtype=float)
            included.append(feature)
            Xcols.append(np.log2(aux / base))
            Ycols.append(np.log2(late / base))
    if not included:
        raise RuntimeError(f"No available features in {filename}")
    return {
        "dataset": label,
        "subjects": complete,
        "features": included,
        "X": np.column_stack(Xcols),
        "Y": np.column_stack(Ycols),
        "availability": availability,
        "structure": {"rows_total": len(rows), "ids_total": len({r["id"] for r in rows}),
                      "selected_visit_rows": dict(counts), "complete_triplets": len(complete),
                      "excluded_fully_unmeasured_triplets": len(by_subject) - len(complete),
                      "time_mapping": {"1": "baseline", "7": "120", "8": "180"}},
    }


def st_source(quant_file, factors_file, scenario, labels):
    factor_path, quant_path = SOURCE / factors_file, SOURCE / quant_file
    factors = json.loads(factor_path.read_text(encoding="utf-8"))
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    sample_meta = {}
    subject_visit = defaultdict(dict)
    for factor_id, record in factors.items():
        local_id = record.get("local_sample_id")
        visit = record.get("factors", "").split(":", 1)[-1].strip().lower()
        if local_id:
            sample_meta[local_id] = {"factor_id": factor_id, "visit": visit}
            sid = local_id.split("_", 1)[0]
            if visit in labels:
                if visit in subject_visit[sid]:
                    raise RuntimeError(f"Duplicate ST003348 subject/visit {sid}/{visit}")
                subject_visit[sid][visit] = local_id
    if not quant:
        raise RuntimeError("ST003348 quant file contains no metabolites")
    factor_sample_ids = set(sample_meta)
    quant_sample_ids = set(quant[next(iter(quant))].get("DATA", {}))
    if factor_sample_ids != quant_sample_ids:
        raise RuntimeError(
            f"ST003348 factor/quant sample mapping mismatch: factors={len(factor_sample_ids)}, "
            f"quant={len(quant_sample_ids)}, missing={len(factor_sample_ids - quant_sample_ids)}, "
            f"extra={len(quant_sample_ids - factor_sample_ids)}"
        )
    selected_subjects = sorted(sid for sid, visits in subject_visit.items()
                               if all(v in visits for v in labels))
    required_samples = [subject_visit[sid][visit] for sid in selected_subjects for visit in labels]
    included, availability, Xcols, Ycols = [], [], [], []
    for metabolite_id, record in quant.items():
        data = record.get("DATA", {})
        name = record.get("metabolite_name") or record.get("refmet_name") or str(metabolite_id)
        feature = f"{metabolite_id}:{name}"
        reason = "included"
        values = {visit: [] for visit in labels}
        for sid in selected_subjects:
            for visit in labels:
                local_id = subject_visit[sid][visit]
                value = as_float(data.get(local_id))
                if value is None:
                    reason = "excluded: missing/non-finite at required visit"
                    break
                if value <= 0:
                    reason = "excluded: non-positive concentration prevents log2 change"
                    break
                values[visit].append(value)
            if reason != "included":
                break
        availability.append({"dataset": scenario, "feature": feature, "status": reason})
        if reason == "included":
            base = np.asarray(values[labels[0]], dtype=float)
            aux = np.asarray(values[labels[1]], dtype=float)
            late = np.asarray(values[labels[2]], dtype=float)
            included.append(feature)
            Xcols.append(np.log2(aux / base))
            Ycols.append(np.log2(late / base))
    if not included:
        raise RuntimeError(f"No available ST003348 features for {scenario}")
    return {
        "dataset": scenario,
        "subjects": selected_subjects,
        "features": included,
        "X": np.column_stack(Xcols),
        "Y": np.column_stack(Ycols),
        "availability": availability,
        "structure": {"factor_rows": len(factors), "quant_features": len(quant),
                      "subjects_complete": len(selected_subjects),
                      "factor_quant_sample_mapping_exact": True,
                      "factor_quant_sample_count": len(factor_sample_ids),
                      "visit_counts_all_factors": dict(Counter(m["visit"] for m in sample_meta.values())),
                      "selected_visits": {"baseline": labels[0], "auxiliary": labels[1], "target": labels[2]},
                      "unique_selected_samples": len(required_samples)},
    }


def correlation_squares(x, y):
    xc = x - np.mean(x, axis=0)
    yc = y - np.mean(y, axis=0)
    cov = np.sum(xc * yc, axis=0)
    xx = np.sum(xc * xc, axis=0)
    yy = np.sum(yc * yc, axis=0)
    denom = np.sqrt(xx * yy)
    rho = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    rho = np.clip(rho, -1.0, 1.0)
    return rho * rho, rho


def variance_threshold(delta_v):
    c = 1.0 + delta_v
    return (3.0 * c + 2.0 + math.sqrt(8.0 * c * (c + 1.0))) / ((c + 2.0) ** 2)


def predicted_allocation(r, H, delta_v, g):
    m = PILOT_M
    total_budget = m * (3.0 + delta_v) + g * (2.0 + delta_v)
    post_pilot_budget = total_budget - m * (3.0 + delta_v)
    n_gold = min(H, int(math.floor(post_pilot_budget / (2.0 + delta_v) + 1e-9)))
    if n_gold < 1:
        raise RuntimeError(f"No feasible matched-gold allocation for H={H}, g={g}, dv={delta_v}")
    n_no_pilot = min(H, int(math.floor(total_budget / (2.0 + delta_v) + 1e-9)))
    c = 1.0 + delta_v
    candidates = []
    for n in range(4, H + 1):
        for k in range(4, n + 1):
            cost = 2.0 * n + c * k
            if cost <= post_pilot_budget + 1e-10:
                raw_v = (1.0 - r) / k + r / n
                # The finite-pool correction is common to both plans: subtract 1/H.
                v = max(0.0, raw_v - 1.0 / H)
                candidates.append((v, n, k, cost, raw_v))
    if candidates:
        best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        borrow_v, n_borrow, k_borrow, borrow_cost, raw_borrow_v = best
    else:
        borrow_v, n_borrow, k_borrow, borrow_cost, raw_borrow_v = (float("inf"), 0, 0, 0.0, float("inf"))
    gold_v = max(0.0, 1.0 / n_gold - 1.0 / H)
    decision = "borrow" if borrow_v < gold_v - 1e-12 else "gold"
    return {
        "total_budget": total_budget,
        "post_pilot_budget": post_pilot_budget,
        "n_gold": n_gold,
        "n_no_pilot": n_no_pilot,
        "n_borrow": n_borrow,
        "k_borrow": k_borrow,
        "borrow_cost": borrow_cost,
        "borrow_predicted_variance": borrow_v,
        "borrow_predicted_variance_uncorrected": raw_borrow_v,
        "gold_predicted_variance": gold_v,
        "decision": decision,
    }


def score(estimates, truth, scale):
    e = (estimates - truth[None, :]) / scale[None, :]
    mse = np.mean(e * e, axis=1)
    bias = np.mean(e, axis=1)
    rmse = np.sqrt(mse)
    return {"mse": float(np.mean(mse)), "rmse": float(math.sqrt(float(np.mean(mse)))),
            "bias": float(np.mean(bias))}


def simulate_split(data, pilot_idx, held_idx, delta_v, g, rng):
    x_all, y_all = data["X"], data["Y"]
    xp, yp = x_all[pilot_idx], y_all[pilot_idx]
    xh, yh = x_all[held_idx], y_all[held_idx]
    H, F = yh.shape
    if H < PILOT_M or H < 4:
        raise RuntimeError(f"Insufficient held-out subjects: {H}")
    r2, rho = correlation_squares(xp, yp)
    r = float(np.mean(r2))
    mu_x = np.mean(xp, axis=0)
    mu_y = np.mean(yp, axis=0)
    sx = np.std(xp, axis=0, ddof=1)
    sy = np.std(yp, axis=0, ddof=1)
    sx = np.where(sx > 0, sx, 1.0)
    sy = np.where(sy > 0, sy, 1.0)
    xh_z = (xh - mu_x[None, :]) / sx[None, :]
    yh_z = (yh - mu_y[None, :]) / sy[None, :]
    pred = xh_z * rho[None, :]
    residual = yh_z - pred
    truth_z = np.mean(yh_z, axis=0)
    full_sd = np.std(y_all, axis=0, ddof=1)
    full_sd = np.where(full_sd > 0, full_sd, 1.0)
    score_scale_z = full_sd / sy
    alloc = predicted_allocation(r, H, delta_v, g)
    if alloc["n_borrow"] < 1:
        raise RuntimeError(f"No feasible borrowing allocation for H={H}, g={g}, dv={delta_v}")
    perm = np.argsort(rng.random((N_MASKS, H)), axis=1)
    g_idx = perm[:, :alloc["n_gold"]]
    b_n_idx = perm[:, :alloc["n_borrow"]]
    b_k_idx = perm[:, :alloc["k_borrow"]]
    np_idx = perm[:, :alloc["n_no_pilot"]]
    est_gold = np.mean(yh_z[g_idx], axis=1)
    est_borrow = np.mean(pred[b_n_idx], axis=1) + np.mean(residual[b_k_idx], axis=1)
    raw_truth = np.mean(yh, axis=0)
    est_no_pilot = np.mean(yh[np_idx], axis=1)
    scored = {
        "matched_gold": score(est_gold, truth_z, score_scale_z),
        "forced_borrow": score(est_borrow, truth_z, score_scale_z),
        "no_pilot_gold": score(est_no_pilot, raw_truth, full_sd),
    }
    scored["threshold_policy"] = scored["forced_borrow"] if alloc["decision"] == "borrow" else scored["matched_gold"]
    realized_borrow_better = scored["forced_borrow"]["mse"] < scored["matched_gold"]["mse"]
    decision_agreement = ((alloc["decision"] == "borrow") == realized_borrow_better)
    out = {
        "mean_pilot_r2": r,
        "r_star_display": variance_threshold(delta_v),
        "decision": alloc["decision"],
        "decision_agreement": int(decision_agreement),
        "n_eval": H,
        "n_features": F,
        "n_gold": alloc["n_gold"],
        "n_no_pilot": alloc["n_no_pilot"],
        "n_borrow": alloc["n_borrow"],
        "k_borrow": alloc["k_borrow"],
        "borrow_predicted_variance": alloc["borrow_predicted_variance"],
        "gold_predicted_variance": alloc["gold_predicted_variance"],
        "borrow_cost": alloc["borrow_cost"],
        "post_pilot_budget": alloc["post_pilot_budget"],
        "benefit_gold_minus_borrow_mse": scored["matched_gold"]["mse"] - scored["forced_borrow"]["mse"],
        "scores": scored,
    }
    return out


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hashes = verify_sources()
    pomet_primary = pomet_source("pomet_205866.csv", "PoMet_205866_primary")
    if len(pomet_primary["subjects"]) != 33:
        raise RuntimeError(f"Expected 33 complete PoMet triplets; found {len(pomet_primary['subjects'])}")
    pomet_sens = pomet_source("pomet_196764.csv", "PoMet_196764_sensitivity",
                              common_subjects=pomet_primary["subjects"])
    if pomet_sens["subjects"] != pomet_primary["subjects"]:
        raise RuntimeError("PoMet sensitivity subjects do not match primary subject set")
    st_main = st_source("ST003348_quant.json", "st003348_factors.json",
                        "ST003348_REST_STAT_REC3_secondary", ("rest", "stat", "rec3"))
    st_new = st_source("ST003348_quant.json", "st003348_factors.json",
                       "ST003348_REST_REC3_REC22_exploratory_new_visit", ("rest", "rec3", "rec22"))
    if len(st_main["subjects"]) != 19 or len(st_new["subjects"]) != 19:
        raise RuntimeError("Expected 19 mapped ST003348 subjects for each complete visit scenario")
    datasets = [pomet_primary, pomet_sens, st_main, st_new]
    dataset_by_name = {d["dataset"]: d for d in datasets}

    # Construct paired subject splits once per source family. PoMet sensitivity
    # shares the exact split with primary; ST003348 visit scenarios also share it.
    split_rng = np.random.default_rng(SEED)
    pomet_splits, st_splits = [], []
    for _ in range(N_OUTER):
        pomet_splits.append(split_rng.permutation(len(pomet_primary["subjects"])))
        st_splits.append(split_rng.permutation(len(st_main["subjects"])))

    structures = {d["dataset"]: d["structure"] for d in datasets}
    protocol = {
        "frozen_before_outcomes": True,
        "source_hashes_sha256": hashes,
        "seed": SEED,
        "outer_pilot_heldout_splits": N_OUTER,
        "inner_finite_pool_random_masks_per_split": N_MASKS,
        "pilot_m": PILOT_M,
        "g_values": list(G_VALUES),
        "delta_v_grid": list(DELTA_V),
        "budget": "B=m(3+dv)+g(2+dv); post-pilot R=g(2+dv)",
        "gold_cost": "(2+dv)N; N_gold=floor(R/(2+dv)), capped at held-out H",
        "borrow_cost": "2N+(1+dv)k; k>=4; N<=H; k<=N",
        "borrow_objective": "minimize ((1-r)/k+r/N)-1/H over feasible integer N,k",
        "panel_r": "unweighted mean over all predeclared available marker features of pilot Pearson r_j squared between baseline-to-aux and baseline-to-target log2 changes",
        "slope": "feature-specific pilot Pearson correlation on pilot-standardized changes; frozen within each trial",
        "borrow_estimator": "mean predicted target over N auxiliary subjects plus mean paired residual over nested k gold subjects",
        "discrete_policy": "borrow iff minimum finite-pool predicted variance is strictly lower than matched-gold predicted variance 1/N_gold-1/H; otherwise gold",
        "continuous_threshold_display_only": "r_star=(3c+2+sqrt(8c(c+1)))/(c+2)^2, c=1+dv",
        "no_pilot_gold": "full B spent on gold subjects, capped at held-out H; no pilot outcomes used in its estimate",
        "evaluation": "without-replacement nested random masks from each held-out pool; target is that split's full held-out mean; pilot subjects are disjoint",
        "standardized_metrics": "per-marker estimation errors divided by that marker's full-source SD of target log2 change, then equally averaged across markers; that SD is used only for scoring, never for allocation or prediction",
        "uncertainty": "2.5th-97.5th percentiles across repeated outer pilot/held-out splits (split variation, not population-level confidence intervals)",
        "coverage": "omitted: finite-pool masks are scored directly against known held-out means; no analytic inferential SE model is claimed",
        "feature_rule": "all concentration features except id/sex/time/glucose/insulin; exclude subjects with a fully unmeasured required visit; retain only features with finite positive values at all required visits in remaining complete subjects, solely to define log2 changes; no correlation or test-outcome feature selection",
        "structures": structures,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    avail = []
    for d in datasets:
        avail.extend(d["availability"])
    write_csv(OUT / "feature_availability.csv", avail)

    split_rows = []
    loop_started = time.time()
    datasets_paired = [
        (pomet_primary, pomet_splits),
        (pomet_sens, pomet_splits),
        (st_main, st_splits),
        (st_new, st_splits),
    ]
    for di, (data, splits) in enumerate(datasets_paired, start=1):
        print(f"START dataset={data['dataset']} subjects={len(data['subjects'])} features={len(data['features'])}", flush=True)
        for split_no, perm in enumerate(splits, start=1):
            pilot_idx = perm[:PILOT_M]
            held_idx = perm[PILOT_M:]
            if set(pilot_idx).intersection(set(held_idx)):
                raise RuntimeError("Pilot/held-out subjects overlap")
            for dv in DELTA_V:
                for g in G_VALUES:
                    rng = np.random.default_rng(SEED + di * 1_000_000 + split_no * 1000 + int(dv * 100) * 10 + g)
                    result = simulate_split(data, pilot_idx, held_idx, dv, g, rng)
                    base = {
                        "dataset": data["dataset"],
                        "scenario": data["dataset"],
                        "outer_split": split_no,
                        "delta_v": dv,
                        "g": g,
                        "n_eval": result["n_eval"],
                        "n_features": result["n_features"],
                        "mean_pilot_r2": result["mean_pilot_r2"],
                        "r_star_display": result["r_star_display"],
                        "policy_decision": result["decision"],
                        "decision_agreement": result["decision_agreement"],
                        "n_gold": result["n_gold"],
                        "n_no_pilot": result["n_no_pilot"],
                        "n_borrow": result["n_borrow"],
                        "k_borrow": result["k_borrow"],
                        "borrow_predicted_variance": result["borrow_predicted_variance"],
                        "gold_predicted_variance": result["gold_predicted_variance"],
                        "borrow_cost": result["borrow_cost"],
                        "post_pilot_budget": result["post_pilot_budget"],
                        "benefit_gold_minus_borrow_mse": result["benefit_gold_minus_borrow_mse"],
                    }
                    for strategy, metrics in result["scores"].items():
                        split_rows.append({**base, "strategy": strategy, **metrics})
            if split_no % 100 == 0:
                elapsed = time.time() - loop_started
                print(f"PROGRESS dataset={data['dataset']} splits={split_no}/{N_OUTER} elapsed_s={elapsed:.1f}", flush=True)

    write_csv(OUT / "split_metrics.csv", split_rows)
    summary_rows = []
    keys = sorted({(r["dataset"], r["delta_v"], r["g"], r["strategy"]) for r in split_rows})
    for dataset, dv, g, strategy in keys:
        subset = [r for r in split_rows if r["dataset"] == dataset and r["delta_v"] == dv
                  and r["g"] == g and r["strategy"] == strategy]
        mses = [r["mse"] for r in subset]
        rmses = [r["rmse"] for r in subset]
        biases = [r["bias"] for r in subset]
        agreements = [r["decision_agreement"] for r in subset]
        benefits = [r["benefit_gold_minus_borrow_mse"] for r in subset]
        policy_borrow = [1 if r["policy_decision"] == "borrow" else 0 for r in subset]
        summary_rows.append({
            "dataset": dataset, "scenario": dataset, "delta_v": dv, "g": g, "strategy": strategy,
            "outer_splits": len(subset), "n_eval": subset[0]["n_eval"], "n_features": subset[0]["n_features"],
            "mean_pilot_r2": float(np.mean([r["mean_pilot_r2"] for r in subset])),
            "r_star_display": subset[0]["r_star_display"],
            "policy_borrow_fraction": float(np.mean(policy_borrow)),
            "median_n_gold": float(np.median([r["n_gold"] for r in subset])),
            "median_n_no_pilot": float(np.median([r["n_no_pilot"] for r in subset])),
            "median_n_borrow": float(np.median([r["n_borrow"] for r in subset])),
            "median_k_borrow": float(np.median([r["k_borrow"] for r in subset])),
            "mse_mean": float(np.mean(mses)), "mse_split_p025": percentile(mses, .025), "mse_split_p975": percentile(mses, .975),
            "rmse_mean": float(np.mean(rmses)), "rmse_split_p025": percentile(rmses, .025), "rmse_split_p975": percentile(rmses, .975),
            "bias_mean": float(np.mean(biases)), "bias_split_p025": percentile(biases, .025), "bias_split_p975": percentile(biases, .975),
            "decision_agreement_fraction": float(np.mean(agreements)),
            "agreement_split_p025": percentile(agreements, .025), "agreement_split_p975": percentile(agreements, .975),
            "gold_minus_borrow_mse_mean": float(np.mean(benefits)),
            "benefit_split_p025": percentile(benefits, .025), "benefit_split_p975": percentile(benefits, .975),
        })
    write_csv(OUT / "summary.csv", summary_rows)
    interpretation = [
        "External empirical validation: descriptive finite-pool simulation; no COVID-19, disease, or causal inference.",
        f"Source subjects: PoMet 33 matched triplets; ST003348 19 subjects per visit scenario. Outer splits={N_OUTER}, masks per split={N_MASKS}.",
        "Intervals in summary.csv are repeated split-variation percentiles, not population confidence intervals. See summary.csv for all 5 Δv × 2 g conditions.",
    ]
    for dataset in sorted({r["dataset"] for r in summary_rows}):
        panel = [r for r in summary_rows if r["dataset"] == dataset]
        policy_rows = [r for r in panel if r["strategy"] == "threshold_policy"]
        borrow_rows = [r for r in panel if r["strategy"] == "forced_borrow"]
        borrow_wins = sum(r["gold_minus_borrow_mse_mean"] > 0 for r in borrow_rows)
        policy_fraction = [r["policy_borrow_fraction"] for r in policy_rows]
        policy_mse = [r["mse_mean"] for r in policy_rows]
        agreement = [r["decision_agreement_fraction"] for r in policy_rows]
        interpretation.append(
            f"{dataset}: policy borrow fraction across cost conditions={min(policy_fraction):.3f}–{max(policy_fraction):.3f}; "
            f"forced borrowing had lower mean MSE than matched gold in {borrow_wins}/{len(borrow_rows)} conditions; "
            f"policy MSE range={min(policy_mse):.4g}–{max(policy_mse):.4g}; "
            f"mean decision agreement across conditions={float(np.mean(agreement)):.3f}."
        )
    interpretation.append("No-pilot gold is capped at the available held-out pool; where its sample size reaches H it is a census reference and has zero finite-pool sampling error by construction.")
    (OUT / "interpretation.txt").write_text("\n".join(interpretation) + "\n", encoding="utf-8")
    meta = {
        "finished_utc_epoch": time.time(), "elapsed_seconds": time.time() - started,
        "source_sha256": hashes, "protocol_sha256": sha256(OUT / "protocol.json"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "outer_splits": N_OUTER, "inner_masks": N_MASKS,
        "source_structures": structures,
        "included_feature_counts": {d["dataset"]: len(d["features"]) for d in datasets},
        "split_counts": len(split_rows),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE elapsed_s={meta['elapsed_seconds']:.1f} split_rows={len(split_rows)} summary_rows={len(summary_rows)}", flush=True)


if __name__ == "__main__":
    main()

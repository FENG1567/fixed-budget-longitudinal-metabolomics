#!/usr/bin/env python3
"""Participant-grouped cross-validation for multiple auxiliary measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEED = 41
N_FOLDS = 5
MIN_TRAIN_FINITE_FRACTION = 0.80


def load(study: str, parsed: Path):
    y = np.load(parsed / f"ST003{study}_Y.npy").astype(float)
    features = [s.strip() for s in (parsed / f"ST003{study}_features.txt").read_text(encoding="utf-8").splitlines()]
    rows = []
    with (parsed / f"ST003{study}_samples.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({"id": row["sample_id"], "t": int(row["time"]),
                         "ep": row["episode"], "subj": row["subject"]})
    if len(rows) != y.shape[1] or len(features) != y.shape[0]:
        raise ValueError(f"metadata mismatch for ST003{study}: Y={y.shape}, rows={len(rows)}, features={len(features)}")
    idx = {r["id"]: i for i, r in enumerate(rows)}
    by_time = {}
    for t in sorted({r["t"] for r in rows}):
        sub = sorted([r for r in rows if r["t"] == t], key=lambda r: r["ep"])
        by_time[t] = (np.array([idx[r["id"]] for r in sub], dtype=int),
                      np.array([r["subj"] for r in sub], dtype=str))
    base_subj = by_time[0][1]
    for t, (_, subjects) in by_time.items():
        if not np.array_equal(subjects, base_subj):
            raise ValueError(f"episode/subject ordering differs at t={t} for ST003{study}")
    return y, features, base_subj, by_time


def transform(values: np.ndarray, median: float) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, median)
    return np.log2(np.maximum(filled, 0.0) + 1.0)


def run_dataset(study: str, times: list[int], late: int, rng: np.random.Generator, parsed: Path):
    yraw, features, subjects, by_time = load(study, parsed)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)
    folds = [unique_subjects[i::N_FOLDS] for i in range(N_FOLDS)]
    all_times = [0] + times + [late]
    nf, nobs = yraw.shape[0], len(subjects)
    preds = np.full((nf, nobs), np.nan)
    observed = np.full((nf, nobs), np.nan)
    eligible_folds = np.zeros(nf, dtype=int)
    imputed_train = np.zeros(nf, dtype=int)
    imputed_test = np.zeros(nf, dtype=int)

    for held_subjects in folds:
        te = np.isin(subjects, held_subjects)
        tr = ~te
        train_indices = {t: by_time[t][0][tr] for t in all_times}
        test_indices = {t: by_time[t][0][te] for t in all_times}
        for f in range(nf):
            tr_cols = np.concatenate([train_indices[t] for t in all_times])
            finite_train = np.isfinite(yraw[f, tr_cols])
            if finite_train.mean() < MIN_TRAIN_FINITE_FRACTION or finite_train.sum() < 8:
                continue
            medians = {}
            for t in all_times:
                vals = yraw[f, train_indices[t]]
                finite = vals[np.isfinite(vals)]
                if not finite.size:
                    medians = {}
                    break
                medians[t] = float(np.median(finite))
            if not medians:
                continue
            ztr = {t: transform(yraw[f, train_indices[t]], medians[t]) for t in all_times}
            zte = {t: transform(yraw[f, test_indices[t]], medians[t]) for t in all_times}
            imputed_train[f] += int(sum(np.isnan(yraw[f, train_indices[t]]).sum() for t in all_times))
            imputed_test[f] += int(sum(np.isnan(yraw[f, test_indices[t]]).sum() for t in all_times))
            xtr = np.column_stack([ztr[t] - ztr[0] for t in times])
            xte = np.column_stack([zte[t] - zte[0] for t in times])
            ytr = ztr[late] - ztr[0]
            yte = zte[late] - zte[0]
            design_tr = np.column_stack([xtr, np.ones(len(xtr))])
            coef, *_ = np.linalg.lstsq(design_tr, ytr, rcond=None)
            design_te = np.column_stack([xte, np.ones(len(xte))])
            preds[f, te] = design_te @ coef
            observed[f, te] = yte
            eligible_folds[f] += 1

    feature_rows = []
    for f, feature in enumerate(features):
        complete = (eligible_folds[f] == N_FOLDS and np.isfinite(preds[f]).all()
                    and np.isfinite(observed[f]).all())
        if not complete:
            feature_rows.append({"study": study, "feature": feature, "eligible_all_folds": False,
                                 "n_folds_eligible": int(eligible_folds[f]), "n_oof": int(np.isfinite(preds[f]).sum()),
                                 "predictive_R2": "", "squared_correlation": "",
                                 "train_medians_imputed": int(imputed_train[f]),
                                 "heldout_medians_imputed": int(imputed_test[f])})
            continue
        obs, pred = observed[f], preds[f]
        sst = float(np.sum((obs - obs.mean()) ** 2))
        pred_r2 = float(1.0 - np.sum((obs - pred) ** 2) / sst) if sst > 1e-15 else float("nan")
        corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 1e-12 and np.std(pred) > 1e-12 else float("nan")
        corr2 = corr * corr if np.isfinite(corr) else float("nan")
        feature_rows.append({"study": study, "feature": feature, "eligible_all_folds": True,
                             "n_folds_eligible": int(eligible_folds[f]), "n_oof": int(nobs),
                             "predictive_R2": pred_r2, "squared_correlation": corr2,
                             "train_medians_imputed": int(imputed_train[f]),
                             "heldout_medians_imputed": int(imputed_test[f])})
    scored = [r for r in feature_rows if r["eligible_all_folds"]
              and np.isfinite(r["predictive_R2"]) and np.isfinite(r["squared_correlation"])]
    def summary(key):
        vals = np.array([r[key] for r in scored], dtype=float)
        return {"median": float(np.median(vals)), "q90": float(np.quantile(vals, 0.90))} if len(vals) else {"median": None, "q90": None}
    n_unique = len(unique_subjects)
    results = {"study": study, "times": times, "late": late, "subjects": n_unique,
               "episodes_rows": nobs, "features_loaded": nf,
               "features_eligible_all_folds": int(sum(r["eligible_all_folds"] for r in feature_rows)),
               "features_scored": len(scored), "predictive_R2": summary("predictive_R2"),
               "squared_correlation": summary("squared_correlation"),
               "folds": N_FOLDS, "fold_assignment": "subject-grouped, shuffled once with seed 41",
               "eligibility": f"training-fold finite fraction >= {MIN_TRAIN_FINITE_FRACTION:.0%} across baseline, auxiliaries, and late outcome",
               "imputation": "nonfinite abundance values replaced by feature x timepoint median estimated from training subjects only; transform log2(x+1)",
               "n_train_imputed_total": int(imputed_train.sum()), "n_heldout_imputed_total": int(imputed_test.sum())}
    return results, feature_rows


def main(parsed, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_results, all_features = [], []
    for study, times, late in (("894", [30, 60, 120], 180), ("895", [30, 60, 120], 180), ("896", [60], 120)):
        result, features = run_dataset(study, times, late, rng, parsed)
        all_results.append(result)
        all_features.extend(features)
        print(f"ST003{study}: m={len(times)}, subjects={result['subjects']}, "
              f"eligible/scored={result['features_eligible_all_folds']}/{result['features_scored']}; "
              f"predictive R2 median/q90={result['predictive_R2']['median']:.3f}/{result['predictive_R2']['q90']:.3f}; "
              f"corr2 median/q90={result['squared_correlation']['median']:.3f}/{result['squared_correlation']['q90']:.3f}; "
              f"imputed train/test={result['n_train_imputed_total']}/{result['n_heldout_imputed_total']}")
    with (out / "multi_aux_metrics_by_study.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    with (out / "multi_aux_metrics_by_feature.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_features[0].keys()))
        w.writeheader()
        w.writerows(all_features)
    (out / "multi_aux_metrics.json").write_text(json.dumps({"seed": SEED, "n_folds": N_FOLDS,
        "no_outcome_tuning": True, "metrics": all_results}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote outputs under {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.parsed, args.out)

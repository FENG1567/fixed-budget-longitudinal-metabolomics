#!/usr/bin/env python3
"""Reproduce the three exploratory panel checks reported in Supplement S3.

This is a portability-only extraction of the archived producers B0_panel_local.py,
B4_stability.py, and B6_loso.py. The calculations, transformations, fold/split
seeds, feature filters, and top-20 rules are retained for the reported checks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_study(parsed: Path, study: str):
    y = np.load(parsed / f"ST003{study}_Y.npy").astype(float)
    features = [line.strip() for line in (parsed / f"ST003{study}_features.txt").open(encoding="utf-8")]
    keep = y.min(axis=1) > 0
    y = y[keep]
    features = [feature for feature, retain in zip(features, keep) if retain]
    rows = []
    with (parsed / f"ST003{study}_samples.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"id": row["sample_id"], "time": int(row["time"]),
                         "episode": row["episode"], "subject": row["subject"]})
    return y, features, rows


def matrices(parsed: Path, study: str, early: int, late: int):
    y, features, rows = load_study(parsed, study)
    index = {row["id"]: i for i, row in enumerate(rows)}

    def columns(time):
        selected = [row for row in rows if row["time"] == time]
        selected.sort(key=lambda row: row["episode"])
        return ([index[row["id"]] for row in selected],
                [row["episode"] for row in selected],
                [row["subject"] for row in selected])

    c0, e0, subjects = columns(0)
    c_early, e_early, _ = columns(early)
    c_late, e_late, _ = columns(late)
    assert e0 == e_early == e_late, "time-point episode order differs"
    baseline = np.log2(y[:, c0] + 1)
    early_change = np.log2(y[:, c_early] + 1) - baseline
    late_change = np.log2(y[:, c_late] + 1) - baseline
    return early_change, late_change, np.asarray(subjects), features


def r2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 6:
        return np.nan
    a, b = a[valid], b[valid]
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else np.nan


def folds(subjects, rng, k=5):
    unique = np.unique(subjects)
    rng.shuffle(unique)
    return [unique[i::k] for i in range(k)]


def cv_topk(early, late, subjects, rng, k_top=20, k=5):
    score_early = np.full(early.shape[1], np.nan)
    score_late = np.full(late.shape[1], np.nan)
    for held in folds(subjects, rng, k):
        train = ~np.isin(subjects, held)
        test = np.isin(subjects, held)
        feature_r = np.array([abs(r2(early[f, train], late[f, train]))
                              for f in range(early.shape[0])])
        feature_r = np.nan_to_num(feature_r, nan=0.0)
        top = np.argsort(-feature_r)[:k_top]
        score_early[test] = np.nanmean(early[np.ix_(top, np.where(test)[0])], axis=0)
        score_late[test] = np.nanmean(late[np.ix_(top, np.where(test)[0])], axis=0)
    valid = np.isfinite(score_early) & np.isfinite(score_late)
    return r2(score_early[valid], score_late[valid]) ** 2


def cv_pc1(early, late, subjects, rng, k=5):
    score_early = np.full(early.shape[1], np.nan)
    score_late = np.full(late.shape[1], np.nan)
    for held in folds(subjects, rng, k):
        train = ~np.isin(subjects, held)
        test = np.isin(subjects, held)
        x = early[:, train]
        x = x - np.nanmean(x, axis=1, keepdims=True)
        x = np.nan_to_num(x, nan=0.0)
        u, _, _ = np.linalg.svd(x, full_matrices=False)
        first_component = u[:, 0]
        score_early[test] = first_component @ early[:, test]
        score_late[test] = first_component @ late[:, test]
    valid = np.isfinite(score_early) & np.isfinite(score_late)
    return r2(score_early[valid], score_late[valid]) ** 2


def five_fold(parsed: Path):
    early, late, subjects, _ = matrices(parsed, "894", 120, 180)
    rng = np.random.default_rng(11)
    # B2_panel_k invokes B0's fold helpers in this order. Its earlier top-5 and
    # top-10 calls consume two shuffles before the reported top-20 call.
    cv_pc1(early, late, subjects, rng, 5)
    cv_topk(early, late, subjects, rng, 5, 5)
    cv_topk(early, late, subjects, rng, 10, 5)
    return float(cv_topk(early, late, subjects, rng, 20, 5))


def half_split(parsed: Path):
    early, late, subjects, _ = matrices(parsed, "894", 120, 180)
    rng = np.random.default_rng(31)

    def split_halves():
        unique = np.unique(subjects)
        rng.shuffle(unique)
        half = len(unique) // 2
        return unique[:half], unique[half:]

    def top_set(columns):
        correlations = np.array([r2(early[f, columns], late[f, columns])
                                 for f in range(early.shape[0])])
        correlations = np.nan_to_num(correlations, nan=0.0)
        return set(np.argsort(-np.abs(correlations))[:20].tolist())

    held_scores, similarities = [], []
    for _ in range(40):
        a, b = split_halves()
        cols_a = np.where(np.isin(subjects, a))[0]
        cols_b = np.where(np.isin(subjects, b))[0]
        selected_a = top_set(cols_a)
        selected_b = top_set(cols_b)
        similarities.append(len(selected_a & selected_b) / len(selected_a | selected_b))
        selected_rows = list(selected_a)
        score_early = np.nanmean(early[np.ix_(selected_rows, cols_b)], axis=0)
        score_late = np.nanmean(late[np.ix_(selected_rows, cols_b)], axis=0)
        valid = np.isfinite(score_early) & np.isfinite(score_late)
        held_scores.append(r2(score_early[valid], score_late[valid]) ** 2)
    return float(np.mean(held_scores)), float(np.mean(similarities))


def cross_study(parsed: Path):
    early_5, late_5, _, features_5 = matrices(parsed, "895", 120, 180)
    early_6, late_6, _, features_6 = matrices(parsed, "896", 60, 120)
    shared = sorted(set(features_5) & set(features_6))
    if len(shared) < 60:
        raise ValueError(f"Only {len(shared)} jointly eligible features; archived producer requires 60")
    index_5 = {name: i for i, name in enumerate(features_5)}
    index_6 = {name: i for i, name in enumerate(features_6)}
    common_5 = np.array([index_5[name] for name in shared])
    common_6 = np.array([index_6[name] for name in shared])
    top_5_r = np.array([r2(early_5[f], late_5[f]) for f in common_5])
    top_5_r = np.nan_to_num(top_5_r, nan=0.0)
    selected_5 = common_5[np.argsort(-np.abs(top_5_r))[:20]]
    selected_names_5 = [features_5[i] for i in selected_5]
    target_6 = np.array([index_6[name] for name in selected_names_5])
    pred_early_6 = np.nanmean(early_6[target_6], axis=0)
    pred_late_6 = np.nanmean(late_6[target_6], axis=0)
    valid_6 = np.isfinite(pred_early_6) & np.isfinite(pred_late_6)
    transfer_5_to_6 = r2(pred_early_6[valid_6], pred_late_6[valid_6]) ** 2

    top_6_r = np.array([r2(early_6[f], late_6[f]) for f in common_6])
    top_6_r = np.nan_to_num(top_6_r, nan=0.0)
    selected_6 = common_6[np.argsort(-np.abs(top_6_r))[:20]]
    selected_names_6 = [features_6[i] for i in selected_6]
    target_5 = np.array([index_5[name] for name in selected_names_6])
    pred_early_5 = np.nanmean(early_5[target_5], axis=0)
    pred_late_5 = np.nanmean(late_5[target_5], axis=0)
    valid_5 = np.isfinite(pred_early_5) & np.isfinite(pred_late_5)
    transfer_6_to_5 = r2(pred_early_5[valid_5], pred_late_5[valid_5]) ** 2
    return len(shared), float(transfer_5_to_6), float(transfer_6_to_5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", type=Path, required=True,
                        help="Input directory containing the ST003894/5/6 parsed matrices")
    parser.add_argument("--out", type=Path, required=True,
                        help="New output directory; no source data are copied there")
    args = parser.parse_args()
    parsed, out = args.parsed.resolve(), args.out.resolve()
    if not parsed.is_dir():
        parser.error(f"Parsed input directory does not exist: {parsed}")
    if out == parsed or parsed in out.parents:
        parser.error("Output must be outside the input directory")
    out.mkdir(parents=True, exist_ok=False)

    five_fold_r2 = five_fold(parsed)
    half_r2, half_jaccard = half_split(parsed)
    shared_count, transfer_5_to_6, transfer_6_to_5 = cross_study(parsed)
    results = {
        "five_fold_ST003894_top20_rho2": five_fold_r2,
        "half_split_ST003894_top20_mean_rho2": half_r2,
        "half_split_ST003894_top20_mean_jaccard": half_jaccard,
        "cross_study_shared_feature_count": shared_count,
        "cross_study_ST003895_to_ST003896_top20_rho2": transfer_5_to_6,
        "cross_study_ST003896_to_ST003895_top20_rho2": transfer_6_to_5,
    }
    (out / "panel_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (out / "panel_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "value"])
        writer.writerows(results.items())
    print(json.dumps(results, indent=2))
    print(f"wrote {out / 'panel_results.csv'} and {out / 'panel_results.json'}")


if __name__ == "__main__":
    main()

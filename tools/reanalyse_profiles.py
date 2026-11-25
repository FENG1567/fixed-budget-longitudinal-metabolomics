#!/usr/bin/env python3
"""Descriptive analysis of early and late changes in the study profiles.

Reads de-identified/public matrices under the selected data root.
Writes aggregate CSVs to --outdir.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "inputs"
PARSED = ROOT / "01_data/parsed"
HUMET = ROOT / "01_data/external/mtbls89"
NBOOT = 2000
SEED = 20261123


def read_rows(path, delimiter=","):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def read_summary_rows(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary_rows(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_tod(value):
    value = str(value).strip()
    if not value or value.lower() in {"n.d.", "na", "nan"}:
        return None
    if ":" in value:
        parts = value.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    if "." in value:
        h, m = value.split(".")
        if len(m) <= 2:
            return float(h) + float(m) / 60.0
        return float(value) * 24.0
    return float(value)


def bh(pvals):
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    ix = np.where(ok)[0]
    order = ix[np.argsort(p[ix])]
    values = p[order] * len(order) / np.arange(1, len(order) + 1)
    q[order] = np.minimum.accumulate(values[::-1])[::-1]
    q[ok] = np.clip(q[ok], 0.0, 1.0)
    return q


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    if not np.isfinite(t) or df <= 0:
        return np.nan
    x = df / (df + float(t) ** 2)
    return float(betai(df / 2.0, 0.5, x))


def transform(values, mode):
    values = np.asarray(values, float)
    if mode == "log2(x+1)":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log2(values + 1.0)
    if mode == "log2(x)":
        out = np.full(values.shape, np.nan, float)
        pos = np.isfinite(values) & (values > 0)
        out[pos] = np.log2(values[pos])
        return out
    raise ValueError(mode)


def feature_change_arrays(base, early, late, mode):
    b = transform(base, mode)
    e = transform(early, mode)
    l = transform(late, mode)
    return e - b, l - b


def feature_stats(d_early, d_late):
    joint = np.isfinite(d_early) & np.isfinite(d_late)
    counts = joint.sum(axis=1)
    n_units = d_early.shape[1]
    enough = counts > 0.8 * n_units  # deliberately strict >80%
    sd_e = np.full(d_early.shape[0], np.nan)
    sd_l = np.full(d_late.shape[0], np.nan)
    for i in np.where(enough)[0]:
        sd_e[i] = np.std(d_early[i, joint[i]], ddof=0)
        sd_l[i] = np.std(d_late[i, joint[i]], ddof=0)
    eligible = enough & (sd_e > 0.05) & (sd_l > 0.05)
    r2 = np.full(d_early.shape[0], np.nan)
    for i in np.where(eligible)[0]:
        x = d_early[i, joint[i]]
        y = d_late[i, joint[i]]
        xc, yc = x - x.mean(), y - y.mean()
        den = math.sqrt(float(np.dot(xc, xc) * np.dot(yc, yc)))
        if den > 0:
            r2[i] = (float(np.dot(xc, yc)) / den) ** 2
    eligible &= np.isfinite(r2)
    return dict(joint=joint, counts=counts, enough=enough, sd_e=sd_e, sd_l=sd_l,
                eligible=eligible, r2=r2)


def cluster_bootstrap_median(d_early, d_late, clusters, eligible, reps, seed):
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in unique}
    boots = []
    feature_counts = []
    for _ in range(reps):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[c] for c in chosen])
        x = d_early[eligible][:, idx]
        y = d_late[eligible][:, idx]
        ok = np.isfinite(x) & np.isfinite(y)
        n = ok.sum(axis=1)
        sx = np.where(ok, x, 0.0).sum(axis=1)
        sy = np.where(ok, y, 0.0).sum(axis=1)
        mx = np.divide(sx, n, out=np.zeros_like(sx), where=n > 0)
        my = np.divide(sy, n, out=np.zeros_like(sy), where=n > 0)
        xc = np.where(ok, x - mx[:, None], 0.0)
        yc = np.where(ok, y - my[:, None], 0.0)
        cross = (xc * yc).sum(axis=1)
        ssx = (xc * xc).sum(axis=1)
        ssy = (yc * yc).sum(axis=1)
        den = np.sqrt(ssx * ssy)
        valid = (n >= 2) & (den > 0)
        r2 = np.full(len(n), np.nan)
        r2[valid] = np.minimum((cross[valid] / den[valid]) ** 2, 1.0)
        if np.isfinite(r2).any():
            boots.append(float(np.nanmedian(r2)))
            feature_counts.append(int(np.isfinite(r2).sum()))
    return np.asarray(boots), np.asarray(feature_counts)


def summarize_profile(profile, source, challenge, early_min, late_min, target,
                      d_early, d_late, clusters, n_subjects, n_visit_units,
                      n_features_input, mode, reps=NBOOT, seed=SEED,
                      n_features_source=None, n_source_samples=None):
    s = feature_stats(d_early, d_late)
    ix = s["eligible"]
    r2 = s["r2"][ix]
    b, bc = cluster_bootstrap_median(d_early, d_late, clusters, ix, reps, seed)
    pair = s["counts"][ix]
    row = dict(profile=profile, source=source, challenge=challenge,
               early_min=early_min, late_min=late_min, target_unit=target,
               transform=mode, n_subjects=n_subjects, n_visit_units=n_visit_units,
               n_source_samples=(int(n_source_samples) if n_source_samples is not None else n_visit_units),
               features_source_total=(int(n_features_source) if n_features_source is not None else n_features_input),
               features_input=n_features_input,
               features_joint_complete_gt80=int(s["enough"].sum()),
               features_both_sd_gt005=int((s["enough"] & (s["sd_e"] > .05) & (s["sd_l"] > .05)).sum()),
               features_retained=int(ix.sum()), pair_count_min=int(pair.min()) if len(pair) else 0,
               pair_count_median=float(np.median(pair)) if len(pair) else np.nan,
               pair_count_max=int(pair.max()) if len(pair) else 0,
               median_rho2=float(np.median(r2)) if len(r2) else np.nan,
               q90_rho2=float(np.quantile(r2, .9)) if len(r2) else np.nan,
               cluster_bootstrap_ci95_low=float(np.quantile(b, .025)) if len(b) else np.nan,
               cluster_bootstrap_ci95_high=float(np.quantile(b, .975)) if len(b) else np.nan,
               bootstrap_reps_requested=reps, bootstrap_reps_completed=len(b),
               bootstrap_features_median=int(np.median(bc)) if len(bc) else 0,
               bootstrap_features_min=int(bc.min()) if len(bc) else 0)
    return row, s


def cluster_test_pvalues(values, clusters):
    """CR1 cluster-robust one-sample t test, finite df = clusters - 1."""
    clusters = np.asarray(clusters)
    u = np.unique(clusters)
    p = np.full(values.shape[0], np.nan)
    means = np.full(values.shape[0], np.nan)
    for f in range(values.shape[0]):
        v = values[f]
        valid = np.isfinite(v)
        if valid.sum() < 2:
            continue
        vv, gg = v[valid], clusters[valid]
        group_sums = []
        for c in np.unique(gg):
            group_sums.append(float(np.sum(vv[gg == c] - np.mean(vv))))
        g = len(group_sums)
        if g < 2:
            continue
        n = len(vv)
        mean = float(np.mean(vv))
        var = (g / (g - 1.0)) * float(np.dot(group_sums, group_sums)) / (n * n)
        if var <= 0:
            p[f] = 0.0 if mean != 0 else 1.0
        else:
            p[f] = t_two_sided_p(mean / math.sqrt(var), g - 1)
        means[f] = mean
    return means, p


def paired_fdr_row(profile, source, early_min, late_min, target, d_early,
                   d_late, clusters, n_features_input, mode):
    joint = np.isfinite(d_early) & np.isfinite(d_late)
    counts = joint.sum(axis=1)
    enough = counts > 0.8 * d_early.shape[1]
    test_e, test_l = d_early.copy(), d_late.copy()
    test_e[~enough] = np.nan
    test_l[~enough] = np.nan
    paired = test_l - test_e
    _, p_e = cluster_test_pvalues(test_e, clusters)
    _, p_l = cluster_test_pvalues(test_l, clusters)
    mean_delta, p_d = cluster_test_pvalues(paired, clusters)
    q_e, q_l, q_d = bh(p_e), bh(p_l), bh(p_d)
    tested = np.isfinite(p_d)
    late_sig = np.isfinite(q_l) & (q_l < .05)
    early_sig = np.isfinite(q_e) & (q_e < .05)
    paired_sig = np.isfinite(q_d) & (q_d < .05)
    row = dict(profile=profile, source=source, early_min=early_min, late_min=late_min,
               target_unit=target, transform=mode, features_input=n_features_input,
               features_joint_complete_gt80=int(enough.sum()),
               pair_count_min=int(counts[enough].min()) if enough.any() else 0,
               pair_count_median=float(np.median(counts[enough])) if enough.any() else np.nan,
               pair_count_max=int(counts[enough].max()) if enough.any() else 0,
               features_tested=int(tested.sum()), early_cluster_FDR_q_lt_005=int(early_sig.sum()),
               late_cluster_FDR_q_lt_005=int(late_sig.sum()),
               late_q_lt_005_early_q_ge_005=int((late_sig & ~early_sig).sum()),
               paired_late_minus_early_FDR_q_lt_005=int(paired_sig.sum()),
               paired_test_fdr_denominator=int(tested.sum()),
               median_paired_late_minus_early_change=float(np.nanmedian(mean_delta[tested])) if tested.any() else np.nan,
               FDR_method="BH within profile/transform over features with >80% joint baseline/early/late completeness; feature-wise subject-cluster CR1 t tests",
               paired_contrast="(late - baseline) - (early - baseline)")
    return row


def load_chinese(study, early, late):
    Y_all = np.load(PARSED / f"{study}_Y.npy").astype(float)
    features = [x.strip() for x in open(PARSED / f"{study}_features.txt", encoding="utf-8")]
    samples = read_rows(PARSED / f"{study}_samples.csv")
    if len(samples) != Y_all.shape[1]:
        raise ValueError(f"{study}: sample metadata columns do not match matrix")
    # Match the declared whole-matrix positivity screen used in the primary analysis.
    positive = np.all(np.isfinite(Y_all) & (Y_all > 0), axis=1)
    Y = Y_all[positive]
    feats = [f for f, keep in zip(features, positive) if keep]
    indices = {}
    for i, row in enumerate(samples):
        key = (row["episode"], int(row["time"]))
        if key in indices:
            raise ValueError(f"{study}: duplicate episode/time {key}")
        indices[key] = i
    episodes = sorted({r["episode"] for r in samples})
    meta_by_ep = {r["episode"]: r for r in samples}
    common = [e for e in episodes if all((e, t) in indices for t in (0, early, late))]
    b = Y[:, [indices[(e, 0)] for e in common]]
    e = Y[:, [indices[(x, early)] for x in common]]
    l = Y[:, [indices[(x, late)] for x in common]]
    subj = np.array([int(meta_by_ep[x]["subject"]) for x in common])
    if study in {"ST003894", "ST003895"}:
        block_values = [meta_by_ep[x]["block"].strip() for x in common]
        if any(not value.isdigit() for value in block_values):
            raise ValueError(f"{study}: expected numeric crossover period for each episode")
        blocks = np.array([int(value) for value in block_values])
    else:
        # The CN-meal dataset has no crossover-period field; it is not used for sensitivity.
        blocks = np.full(len(common), np.nan)
    return dict(Y=Y, Y_all=Y_all, features=feats, all_features=features,
                base=b, early=e, late=l, clusters=subj, blocks=blocks,
                n_subjects=len(np.unique(subj)), n_visit_units=len(common),
                n_source_samples=len(samples), study=study)


def person_mean_arrays(d, mode, early_min, late_min):
    be = transform(d["base"], mode)
    ee = transform(d["early"], mode) - be
    le = transform(d["late"], mode) - be
    subjects = np.unique(d["clusters"])
    eout = np.full((ee.shape[0], len(subjects)), np.nan)
    lout = np.full_like(eout, np.nan)
    for j, subject in enumerate(subjects):
        take = d["clusters"] == subject
        # Mean of within-visit changes; baseline, early, and late remain paired per visit.
        eout[:, j] = np.nanmean(ee[:, take], axis=1)
        lout[:, j] = np.nanmean(le[:, take], axis=1)
    return eout, lout, subjects


def load_humet():
    meta = read_rows(HUMET / "s_MTBLS89.txt", delimiter="\t")
    maf_path = HUMET / "m_MTBLS89_metabolite_profiling_mass_spectrometry_v2_maf.tsv"
    with open(maf_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        hdr = next(reader)
        samples = hdr[21:]
        names, rows = [], []
        for row in reader:
            names.append(row[4] if len(row) > 4 and row[4] else row[0])
            vals = []
            for v in row[21:]:
                v = v.strip()
                if not v or v.lower() in {"0", "na", "nan", "n.d."}:
                    vals.append(np.nan)
                else:
                    try:
                        x = float(v)
                        vals.append(x if x > 0 else np.nan)
                    except ValueError:
                        vals.append(np.nan)
            rows.append(vals)
    V = np.asarray(rows, float)
    if V.shape[1] != len(samples):
        raise ValueError("HuMet MAF sample columns mismatch")
    sidx = {s.strip(): i for i, s in enumerate(samples)}
    out = {}
    for ch in ("OGTT", "OLTT"):
        records = []
        for r in meta:
            if r.get("Factor Value[Challenge]", "").strip() != ch:
                continue
            tod = parse_tod(r.get("Factor Value[Time of Day]", ""))
            if tod is None or r.get("Sample Name", "").strip() not in sidx:
                continue
            records.append(dict(sample=r["Sample Name"].strip(),
                                subject=r["Factor Value[Subject Identifier]"].strip(),
                                tod=tod))
        if not records:
            raise ValueError(f"No HuMet records for {ch}")
        base_tod = min(r["tod"] for r in records)
        for r in records:
            r["minute"] = int(round((r["tod"] - base_tod) * 60.0))
        times = sorted({r["minute"] for r in records})
        subjects = sorted({r["subject"] for r in records}, key=lambda x: int(x) if x.isdigit() else x)
        A_sum = np.zeros((len(names), len(subjects), len(times)), dtype=float)
        A_count = np.zeros((len(names), len(subjects), len(times)), dtype=np.int16)
        subidx = {s: i for i, s in enumerate(subjects)}
        tidx = {t: i for i, t in enumerate(times)}
        for r in records:
            v = V[:, sidx[r["sample"]]]
            j, k = subidx[r["subject"]], tidx[r["minute"]]
            # Technical replicate aggregation on the raw intensity scale, feature-wise.
            ok = np.isfinite(v)
            A_sum[ok, j, k] += v[ok]
            A_count[ok, j, k] += 1
        A = np.divide(A_sum, A_count, out=np.full_like(A_sum, np.nan), where=A_count > 0)
        tlookup = {t: i for i, t in enumerate(times)}
        if not all(t in tlookup for t in (0, 120, 180)):
            raise ValueError(f"{ch}: expected 0/120/180 minute windows; got {times}")
        out[ch] = dict(names=names, A=A, times=times, subjects=subjects,
                       base=A[:, :, tlookup[0]], early=A[:, :, tlookup[120]],
                       late=A[:, :, tlookup[180]], n_samples=len(records))
        print(f"Loaded HuMet {ch}: participants={len(subjects)}, samples={len(records)}, "
              f"features={len(names)}, times={times}", flush=True)
    return out


def run(outdir, root=None, resume=False):
    global ROOT, PARSED, HUMET
    if root is not None:
        ROOT = Path(root)
        PARSED = ROOT / "01_data/parsed"
        HUMET = ROOT / "01_data/external/mtbls89"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"PROCESS_PID={os.getpid()}", flush=True)
    print(f"INPUT_ROOT={ROOT}", flush=True)
    profile_path = outdir / "profile_estimates.csv"
    fdr_path = outdir / "paired_cluster_fdr_summary.csv"
    profile_rows = read_summary_rows(profile_path) if resume else []
    fdr_rows = read_summary_rows(fdr_path) if resume else []
    profile_keys = {(r["profile"], r["target_unit"], r["transform"]) for r in profile_rows}
    fdr_keys = {(r["profile"], r["target_unit"], r["transform"]) for r in fdr_rows}
    cn_configs = [
        ("CN-lipid", "ST003894", 120, 180),
        ("CN-metab", "ST003895", 120, 180),
        ("CN-meal", "ST003896", 60, 120),
    ]
    cn_cache = {}
    for profile, study, early, late in cn_configs:
        d = load_chinese(study, early, late)
        cn_cache[profile] = (d, early, late)
        print(f"Loaded {profile}: global_positive_features={len(d['features'])}/"
              f"{len(d['all_features'])}, participants={d['n_subjects']}, "
              f"visit_units={d['n_visit_units']}, times={early}->{late}", flush=True)
        for k, mode in enumerate(("log2(x+1)", "log2(x)")):
            de, dl = feature_change_arrays(d["base"], d["early"], d["late"], mode)
            target_unit = "visit unit; repeated visits clustered by participant"
            key = (profile, target_unit, mode)
            if key not in profile_keys:
                row, _ = summarize_profile(profile, study, "not separately stratified",
                                           early, late, target_unit,
                                           de, dl, d["clusters"], d["n_subjects"], d["n_visit_units"],
                                           len(d["features"]), mode, seed=SEED + k + len(profile),
                                           n_features_source=len(d["all_features"]),
                                           n_source_samples=d["n_source_samples"])
                profile_rows.append(row)
                profile_keys.add(key)
                write_summary_rows(profile_path, profile_rows)
            else:
                row = next(r for r in profile_rows if (r["profile"], r["target_unit"], r["transform"]) == key)
            fkey = (profile, target_unit, mode)
            if fkey not in fdr_keys:
                fdr_rows.append(paired_fdr_row(profile, study, early, late,
                                               target_unit, de, dl, d["clusters"],
                                               len(d["features"]), mode))
                fdr_keys.add(fkey)
                write_summary_rows(fdr_path, fdr_rows)
            print(f"  {mode}: retained_features={row['features_retained']}, "
                  f"median_rho2={float(row['median_rho2']):.6f}, "
                  f"bootstrap_CI=[{float(row['cluster_bootstrap_ci95_low']):.6f},"
                  f"{float(row['cluster_bootstrap_ci95_high']):.6f}]", flush=True)
        if profile in ("CN-lipid", "CN-metab"):
            for k, mode in enumerate(("log2(x+1)", "log2(x)")):
                de, dl, subjects = person_mean_arrays(d, mode, early, late)
                target_unit = "participant mean change across 4 visits"
                key = (profile, target_unit, mode)
                if key not in profile_keys:
                    row, _ = summarize_profile(profile, study, "not separately stratified",
                                               early, late, target_unit,
                                               de, dl, subjects, len(subjects), len(subjects),
                                               len(d["features"]), mode, seed=SEED + 100 + k + len(profile),
                                               n_features_source=len(d["all_features"]),
                                               n_source_samples=d["n_source_samples"])
                    profile_rows.append(row)
                    profile_keys.add(key)
                    write_summary_rows(profile_path, profile_rows)
                else:
                    row = next(r for r in profile_rows if (r["profile"], r["target_unit"], r["transform"]) == key)
                print(f"  {mode} person-mean sensitivity: retained_features={row['features_retained']}, "
                      f"median_rho2={float(row['median_rho2']):.6f}, "
                      f"CI=[{float(row['cluster_bootstrap_ci95_low']):.6f},"
                      f"{float(row['cluster_bootstrap_ci95_high']):.6f}]", flush=True)

    humet = load_humet()
    for j, (ch, d) in enumerate(humet.items()):
        for k, mode in enumerate(("log2(x+1)", "log2(x)")):
            de, dl = feature_change_arrays(d["base"], d["early"], d["late"], mode)
            target_unit = "participant; baseline/early/late joint complete"
            key = (f"DE-{ch}", target_unit, mode)
            if key not in profile_keys:
                row, _ = summarize_profile(f"DE-{ch}", "MTBLS89", ch, 120, 180,
                                           target_unit, de, dl,
                                           np.arange(len(d["subjects"])), len(d["subjects"]),
                                           len(d["subjects"]), len(d["names"]), mode,
                                           seed=SEED + 200 + j * 10 + k,
                                           n_features_source=len(d["names"]),
                                           n_source_samples=d["n_samples"])
                profile_rows.append(row)
                profile_keys.add(key)
                write_summary_rows(profile_path, profile_rows)
            else:
                row = next(r for r in profile_rows if (r["profile"], r["target_unit"], r["transform"]) == key)
            fkey = (f"DE-{ch}", target_unit, mode)
            if fkey not in fdr_keys:
                fdr_rows.append(paired_fdr_row(f"DE-{ch}", "MTBLS89", 120, 180,
                                               target_unit, de, dl,
                                               np.arange(len(d["subjects"])),
                                               len(d["names"]), mode))
                fdr_keys.add(fkey)
                write_summary_rows(fdr_path, fdr_rows)
            print(f"  DE-{ch} {mode}: retained_features={row['features_retained']}, "
                  f"pair_count={row['pair_count_min']}..{row['pair_count_max']}, "
                  f"median_rho2={float(row['median_rho2']):.6f}, "
                  f"bootstrap_CI=[{float(row['cluster_bootstrap_ci95_low']):.6f},"
                  f"{float(row['cluster_bootstrap_ci95_high']):.6f}]", flush=True)

    write_summary_rows(profile_path, profile_rows)
    write_summary_rows(fdr_path, fdr_rows)
    run_info = dict(status="complete", input_root=str(ROOT), original_files_read_only=True,
                    profiles=["CN-lipid/ST003894", "CN-metab/ST003895", "CN-meal/ST003896",
                              "DE-OGTT/MTBLS89", "DE-OLTT/MTBLS89"],
                    primary_transform="log2(x+1)", sensitivity_transform="log2(x) on positive observations",
                    cluster_bootstrap_replicates=NBOOT, seed=SEED,
                    summary_files=[profile_path.name, fdr_path.name],
                    meta_analysis="not performed")
    (outdir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("Wrote aggregate-only summaries to", outdir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--root", help="input root containing 01_data/parsed and 01_data/external/mtbls89")
    parser.add_argument("--resume", action="store_true", help="keep already-written aggregate rows and complete missing analyses")
    args = parser.parse_args()
    run(Path(args.outdir), args.root, args.resume)

#!/usr/bin/env python3
"""One-shot summary of the raw mzML bin matrix: reliability, rho^2, normalisation, intensity.

Robust to unbalanced samples (episodes missing at some timepoints).
Usage: python3 49_raw_summary.py <894|895>
"""
import argparse, csv, json, os
import numpy as np

def load(study, bins_dir, samples_csv):
    d = bins_dir
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    stems = [os.path.splitext(f)[0] for f in files]
    M = np.stack([np.load(os.path.join(d, f)) for f in files], axis=1).T
    mm = {}
    with open(samples_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mm[r["sample_id"]] = r
    keep = [i for i, s in enumerate(stems) if s in mm]
    return M[keep], [mm[stems[i]] for i in keep]


def agg(M, w):
    n = M.shape[1] // w * w
    return M[:, :n].reshape(M.shape[0], -1, w).sum(2)


def index(samp, t):
    return {s["episode"]: i for i, s in enumerate(samp) if int(s["time"]) == t}


def summary(study, bins_dir, samples_csv, w=10, late=180, output=None):
    M, samp = load(study, bins_dir, samples_csv)
    times = sorted({int(s["time"]) for s in samp})
    if late not in times:
        late = max(times)
    idx = {t: index(samp, t) for t in times}
    common = set(idx[0])
    for t in [late, 120, 60]:
        if t in idx:
            common &= set(idx[t])
    common = sorted(common)
    print(f"{study}: samples={len(samp)} times={times} common episodes={len(common)}", flush=True)
    A = agg(M, w)
    res = {"study": study, "n_samples": len(samp), "n_common": len(common), "width_da": w * 0.01,
           "late": late}

    def mat(t, V):
        return V[[idx[t][e] for e in common]]

    rng = np.random.default_rng(5)
    ep = np.array(common)

    # profile reliability (double-centred cosine)
    L = np.log1p(A)
    L = L - L.mean(0, keepdims=True)
    L = L - L.mean(1, keepdims=True)
    Z = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)
    same = []
    for e in common:
        ii = [i for i, s in enumerate(samp) if s["episode"] == e]
        if len(ii) > 1:
            C = Z[ii] @ Z[ii].T
            same.extend(C[np.triu_indices(len(ii), 1)].tolist())
    diff = []
    while len(diff) < 3000:
        a, b = rng.integers(0, len(samp), 2)
        if samp[a]["episode"] != samp[b]["episode"]:
            diff.append(float(Z[a] @ Z[b]))
    res["same_r_median"] = float(np.median(same))
    res["diff_r_median"] = float(np.median(diff))
    print(f"  same-episode r={res['same_r_median']:.3f}  different={res['diff_r_median']:.3f}")

    def rho2(V, early):
        base = np.log2(mat(0, V) + 1)
        Dl = np.log2(mat(late, V) + 1) - base
        De = np.log2(mat(early, V) + 1) - base
        sel = (De.std(0) > 0.05) & (Dl.std(0) > 0.05)
        zs = (De[:, sel] - De[:, sel].mean(0)) / np.maximum(De[:, sel].std(0), 1e-12)
        zl = (Dl[:, sel] - Dl[:, sel].mean(0)) / np.maximum(Dl[:, sel].std(0), 1e-12)
        r2 = (zs * zl).mean(0) ** 2
        return float(np.nanmedian(r2)), float(np.nanquantile(r2, .9)), int(sel.sum())

    res["rho2"] = {}
    for t in [x for x in (30, 60, 120) if x in times and x != late]:
        m, q, n = rho2(A, t)
        res["rho2"][str(t)] = dict(median=m, q90=q, n=n)
        print(f"  rho2(d{t},d{late}) median={m:.3f} q90={q:.3f} n={n}")

    # normalisation variants
    ref = np.median(A, axis=0); ok = ref > 0
    pqn = np.median(A[:, ok] / ref[ok], axis=1)
    variants = {"raw": A, "tic": A / A.sum(1, keepdims=True) * 1e6,
                "median": A / np.maximum(np.median(A, axis=1, keepdims=True), 1e-9) * 1e4,
                "pqn": A / pqn[:, None] * 1e4}
    res["norm"] = {}
    for k, V in variants.items():
        m, q, n = rho2(V, 120 if 120 in times else 60)
        res["norm"][k] = dict(median=m, q90=q)
        print(f"  norm={k:7s} rho2={m:.3f}")

    # intensity deciles
    inten = np.log2(A.mean(0) + 1)
    qs = np.quantile(inten, np.linspace(0, 1, 11))
    res["by_intensity"] = []
    for k in range(10):
        sel = (inten >= qs[k]) & (inten <= qs[k + 1] if k == 9 else inten < qs[k + 1])
        base = np.log2(mat(0, A) + 1)
        Dl = np.log2(mat(late, A) + 1) - base
        De = np.log2(mat(120 if 120 in times else 60, A) + 1) - base
        sel = sel & (De.std(0) > 0.05) & (Dl.std(0) > 0.05)
        zs = (De[:, sel] - De[:, sel].mean(0)) / np.maximum(De[:, sel].std(0), 1e-12)
        zl = (Dl[:, sel] - Dl[:, sel].mean(0)) / np.maximum(Dl[:, sel].std(0), 1e-12)
        r2 = (zs * zl).mean(0) ** 2
        res["by_intensity"].append(dict(decile=k + 1, n=int(sel.sum()),
                                        median_int=float(np.median(inten[sel])) if sel.any() else None,
                                        rho2=float(np.nanmedian(r2))))
    print("  intensity deciles rho2:", " ".join(f"{d['rho2']:.2f}" for d in res["by_intensity"]))
    output = output or f"raw_summary_{study}.json"
    parent = os.path.dirname(os.path.abspath(output))
    os.makedirs(parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=("894", "895"))
    ap.add_argument("--bins", required=True, help="directory with binned .npy spectra")
    ap.add_argument("--samples", required=True, help="study sample map CSV")
    ap.add_argument("--bin-width", type=int, default=10)
    ap.add_argument("--late", type=int, default=180)
    ap.add_argument("--output", help="JSON output (default: raw_summary_<study>.json)")
    a = ap.parse_args()
    summary(a.study, a.bins, a.samples, w=a.bin_width, late=a.late, output=a.output)

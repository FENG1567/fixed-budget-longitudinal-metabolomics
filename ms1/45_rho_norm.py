#!/usr/bin/env python3
"""rho^2(d120,d180) under three normalisations (raw / TIC / median), 0.1 Da bins."""
import argparse, csv, os
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


def rho(A, times, ep, late=180, early=120):
    i0 = np.where(times == 0)[0]; o = np.argsort(ep[i0]); i0 = i0[o]; e0 = ep[i0]
    il = np.where(times == late)[0]; il = il[np.argsort(ep[il])]
    it = np.where(times == early)[0]; it = it[np.argsort(ep[it])]
    assert list(e0) == list(ep[il]) == list(ep[it])
    base = np.log2(A[i0] + 1)
    Dl, Dt = np.log2(A[il] + 1) - base, np.log2(A[it] + 1) - base
    sel = (Dt.std(0) > 0.05) & (Dl.std(0) > 0.05)
    zs = (Dt[:, sel] - Dt[:, sel].mean(0)) / np.maximum(Dt[:, sel].std(0), 1e-12)
    zl = (Dl[:, sel] - Dl[:, sel].mean(0)) / np.maximum(Dl[:, sel].std(0), 1e-12)
    r2 = (zs * zl).mean(0) ** 2
    return float(np.nanmedian(r2)), float(np.nanquantile(r2, .9)), int(sel.sum())


def main(study, bins_dir, samples_csv, w=10):
    M, samp = load(study, bins_dir, samples_csv)
    A = agg(M, w)
    times = np.array([int(s["time"]) for s in samp])
    ep = np.array([s["episode"] for s in samp])
    variants = {
        "raw": A,
        "tic": A / A.sum(1, keepdims=True) * 1e6,
        "median": A / np.maximum(np.median(A, axis=1, keepdims=True), 1e-9) * 1e4,
        "pqn": None,
    }
    # PQN: median of per-sample ratio to the reference (median profile)
    ref = np.median(A, axis=0)
    ok = ref > 0
    pqn_factor = np.median(A[:, ok] / ref[ok], axis=1)
    variants["pqn"] = A / pqn_factor[:, None] * 1e4
    print(f"{study}: {A.shape[1]} bins @ {w*0.01:.2f} Da; rho2(d120,d{180}) by normalisation")
    for k, V in variants.items():
        r2, q90, n = rho(V, times, ep)
        print(f"  {k:7s} rho2_median={r2:.3f} q90={q90:.3f} n={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=("894", "895"))
    ap.add_argument("--bins", required=True, help="directory with binned .npy spectra")
    ap.add_argument("--samples", required=True, help="study sample map CSV")
    ap.add_argument("--bin-width", type=int, default=10, help="number of 0.01-Da bins to aggregate (default: 10)")
    a = ap.parse_args()
    main(a.study, a.bins, a.samples, a.bin_width)

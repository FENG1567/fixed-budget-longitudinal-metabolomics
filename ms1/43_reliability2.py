#!/usr/bin/env python3
"""Raw-data reliability with proper double centring, and noise-dilution curve vs bin width.

Samples x bins matrix. For widths w in {1,5,10,50} bins (0.01..0.5 Da):
  - double-centre (remove bin mean across samples, then sample mean across bins)
  - same-episode vs different-episode profile correlation
  - rho^2(d_120, d_180) across episodes
"""
import argparse, csv, json, os
import numpy as np

def load(study, bins_dir, samples_csv):
    d = bins_dir
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    stems = [os.path.splitext(f)[0] for f in files]
    M = np.stack([np.load(os.path.join(d, f)) for f in files], axis=1).T  # samples x bins
    mm = {}
    with open(samples_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mm[r["sample_id"]] = r
    keep = [i for i, s in enumerate(stems) if s in mm]
    M = M[keep]; stems = [stems[i] for i in keep]
    samp = [mm[s] for s in stems]
    return M, stems, samp


def agg(M, w):
    n = M.shape[1] // w * w
    return M[:, :n].reshape(M.shape[0], -1, w).sum(2)


def main(study, bins_dir, samples_csv, output=None):
    M, stems, samp = load(study, bins_dir, samples_csv)
    times = np.array([int(s["time"]) for s in samp])
    ep = np.array([s["episode"] for s in samp])
    print(f"{M.shape[0]} samples x {M.shape[1]} bins; times {dict(zip(*np.unique(times, return_counts=True)))}",
          flush=True)
    rng = np.random.default_rng(11)
    res = {"study": study, "widths": {}}
    for w in (1, 5, 10, 50):
        A = agg(M, w)
        L = np.log1p(A)
        L = L - L.mean(0, keepdims=True)      # per bin across samples
        L = L - L.mean(1, keepdims=True)      # per sample across bins
        nrm = np.linalg.norm(L, axis=1, keepdims=True)
        Z = L / np.maximum(nrm, 1e-12)
        same = []
        for e in np.unique(ep):
            ii = np.where(ep == e)[0]
            if len(ii) > 1:
                C = Z[ii] @ Z[ii].T
                same.extend(C[np.triu_indices(len(ii), 1)].tolist())
        diff = []
        while len(diff) < 3000:
            a, b = rng.integers(0, len(stems), 2)
            if ep[a] != ep[b]:
                diff.append(float(Z[a] @ Z[b]))
        # rho^2 of changes, same bin width
        cols = lambda t: np.where(times == t)[0]
        i0 = cols(0)
        late = 180 if (times == 180).any() else 120
        il = cols(late)
        order = np.argsort(ep[i0]); i0s = i0[order]; e0 = ep[i0][order]
        order2 = np.argsort(ep[il]); ils = il[order2]
        assert list(e0) == list(ep[il][order2])
        base = np.log2(A[i0s] + 1)
        Dl = np.log2(A[ils] + 1) - base
        out = dict(same_median=float(np.median(same)), diff_median=float(np.median(diff)),
                   ratio=float(np.median(same) / max(np.median(diff), 1e-9)))
        for t in sorted(set(times.tolist()) - {0, late}):
            it = cols(t); o = np.argsort(ep[it]); its = it[o]
            D = np.log2(A[its] + 1) - base
            nz = (D.std(0) > 0.05) & (Dl.std(0) > 0.05)
            Ds, Ls = D[:, nz], Dl[:, nz]
            zs = (Ds - Ds.mean(0)) / np.maximum(Ds.std(0), 1e-12)
            zl = (Ls - Ls.mean(0)) / np.maximum(Ls.std(0), 1e-12)
            r2 = (zs * zl).mean(0) ** 2
            out[f"rho2_{t}"] = float(np.nanmedian(r2))
            out[f"frac_gt083_{t}"] = float(np.nanmean(r2 > 0.83))
        res["widths"][str(w)] = out
        print(f"w={w:2d} ({w*0.01:.2f} Da): same={out['same_median']:.3f} diff={out['diff_median']:.3f} "
              f"ratio={out['ratio']:.2f} | " +
              " ".join(f"rho2({t})={out[f'rho2_{t}']:.3f}" for t in sorted(set(times.tolist()) - {0, late})),
              flush=True)
    output = output or f"raw_reliability2_{study}.json"
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
    ap.add_argument("--output", help="JSON output (default: raw_reliability2_<study>.json)")
    a = ap.parse_args()
    main(a.study, a.bins, a.samples, a.output)

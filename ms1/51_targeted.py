#!/usr/bin/env python3
"""Targeted check: rho^2(d120,d180) at exact m/z windows of known abundant lipids (POS mode)."""
import argparse, csv, os
import numpy as np

LO, W = 200.0, 0.01

TARGETS = {
    "PC 34:1 [M+H]+": 760.5851,
    "PC 32:0 [M+H]+": 734.5694,
    "TG 52:2 [M+NH4]+": 876.8018,
    "TG 54:3 [M+NH4]+": 902.8174,
    "SM d18:1/16:0 [M+H]+": 703.5749,
    "Cer d18:1/24:0 [M+H]+": 650.6113,
    "LPC 16:0 [M+H]+": 496.3403,
    "CE 18:2 [M+NH4]+": 666.6187,
}


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


def main(study, bins_dir, samples_csv):
    M, samp = load(study, bins_dir, samples_csv)
    ep = np.array([s["episode"] for s in samp])
    times = np.array([int(s["time"]) for s in samp])
    i0 = np.where(times == 0)[0]; i0 = i0[np.argsort(ep[i0])]; e0 = ep[i0]
    il = np.where(times == 180)[0]; il = il[np.argsort(ep[il])]
    it = np.where(times == 120)[0]; it = it[np.argsort(ep[it])]
    common = [e for e in e0 if e in set(ep[il]) and e in set(ep[it])]
    rows = {t: {s["episode"]: i for i, s in enumerate(samp) if int(s["time"]) == t} for t in (0, 120, 180)}
    print(f"{study}: {len(common)} complete episodes; targeted rho2 (120 vs 180)")
    for name, mz in TARGETS.items():
        b0 = int((mz - 0.02 - LO) / W); b1 = int((mz + 0.02 - LO) / W)
        if b0 < 0 or b1 >= M.shape[1]:
            print(f"  {name:24s} out of range"); continue
        v0 = M[[rows[0][e] for e in common], b0:b1].sum(1)
        v1 = M[[rows[120][e] for e in common], b0:b1].sum(1)
        v2 = M[[rows[180][e] for e in common], b0:b1].sum(1)
        d1 = np.log2(v1 + 1) - np.log2(v0 + 1)
        d2 = np.log2(v2 + 1) - np.log2(v0 + 1)
        if d1.std() < 1e-9 or d2.std() < 1e-9:
            print(f"  {name:24s} flat"); continue
        r = float(np.corrcoef(d1, d2)[0, 1])
        print(f"  {name:24s} mz={mz:8.3f} median_int={np.median(np.log2(v0+1)):5.1f} "
              f"r={r:6.3f} rho2={r*r:5.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=("894", "895"))
    ap.add_argument("--bins", required=True, help="directory with binned .npy spectra")
    ap.add_argument("--samples", required=True, help="study sample map CSV")
    a = ap.parse_args()
    main(a.study, a.bins, a.samples)

#!/usr/bin/env python3
"""Parse Metabolomics Workbench matrices into tidy matrices and pairing checks.

Outputs (01_data/parsed/):
  <study>_Y.npy        float32 (n_feat x n_sample), log2(x+1) not yet applied
  <study>_samples.csv  sample_id,time,block,subject,visit,episode
  <study>_features.txt feature names (one per line)
  <study>_qc.json      summary
Stdout: summary tables (also written to 03_outputs/tables/).
"""
import argparse
import csv, json, os, re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

STUDIES = ("ST003894", "ST003895", "ST003896")


def load(study, raw):
    with open(raw / f"{study}_data.txt", encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        fac = next(rd)
        names, rows = [], []
        for row in rd:
            if not row:
                continue
            names.append(row[0].strip())
            vals = []
            for x in row[1:]:
                x = x.strip()
                if x == "" or x.upper() in ("NA", "N/A", "NULL", "NAN"):
                    vals.append("nan")
                else:
                    vals.append(x)
            rows.append(vals)
    ids = [x.strip().lstrip("\ufeff") for x in header[1:]]
    times = []
    for f in fac[1:]:
        m = re.search(r"Time:\s*([0-9.]+)", f)
        times.append(int(float(m.group(1))) if m else None)
    Y = np.array(rows, dtype=np.float64)  # may contain empty -> nan
    return ids, times, names, Y


def parse_ids(study, ids):
    if study == "ST003896":
        pat = re.compile(r"^P(\d+)([A-Z])$")
        out = []
        for s in ids:
            m = pat.match(s)
            out.append((None, int(m.group(1)), m.group(2)) if m else (None, None, None))
        return out
    pat = re.compile(r"^(\d)P(\d+)([A-Z])$")
    out = []
    for s in ids:
        m = pat.match(s)
        out.append((int(m.group(1)), int(m.group(2)), m.group(3)) if m else (None, None, None))
    return out


def main(inputs):
    raw = inputs / "01_data/raw"
    OUT = inputs / "01_data/parsed"
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for study in STUDIES:
        ids, times, names, Y = load(study, raw)
        parsed = parse_ids(study, ids)
        bad = [ids[i] for i, p in enumerate(parsed) if p[0] is None and study != "ST003896" or p[1] is None]
        report.append(f"\n===== {study} =====")
        report.append(f"matrix: {Y.shape[0]} features x {Y.shape[1]} samples; raw file rows={len(names)}")
        report.append(f"unique sample ids: {len(set(ids))}/{len(ids)}; duplicate features: {len(names)-len(set(names))}")
        report.append(f"unparsable ids: {len(bad)} {bad[:5]}")
        report.append("time counts: " + str(dict(sorted(Counter(times).items()))))
        # letter x time
        lt = defaultdict(Counter)
        for (blk, subj, vis), t in zip(parsed, times):
            lt[vis][t] += 1
        report.append("visit-letter x time: " + json.dumps({k: dict(v) for k, v in sorted(lt.items())}))
        # block/subject completeness
        eps = sorted({(b, s) for b, s, v in parsed})
        report.append(f"episodes (block,subject): {len(eps)}")
        by_block = defaultdict(set)
        for b, s in eps:
            by_block[b].add(s)
        for b in sorted(by_block):
            subs = sorted(by_block[b])
            missing = [x for x in range(1, max(subs) + 1) if x not in subs]
            report.append(f"  block {b}: {len(subs)} subjects, range {min(subs)}-{max(subs)}, missing {missing}")
        # each episode has exactly the expected time set
        et = defaultdict(list)
        for (b, s, v), t in zip(parsed, times):
            et[(b, s)].append(t)
        expect = {0, 30, 60, 120, 180} if study != "ST003896" else {0, 60, 120}
        bad_eps = {k: sorted(v) for k, v in et.items() if set(v) != expect or len(v) != len(expect)}
        report.append(f"episodes with unexpected time set: {len(bad_eps)} {list(bad_eps.items())[:3]}")
        # missing / zero
        n_nan = int(np.isnan(Y).sum())
        Yz = np.nan_to_num(Y, nan=0.0)
        frac_zero = float((Yz == 0).mean())
        report.append(f"missing cells: {n_nan} ({n_nan/Y.size:.4%}); zero cells: {frac_zero:.4%}")
        feat_ok = np.mean(Yz > 0, axis=1)
        for thr in (0.5, 0.8, 0.95):
            report.append(f"  features with >{int(thr*100)}% nonzero: {int((feat_ok >= thr).sum())}/{len(feat_ok)}")
        report.append(f"value range (nonzero): {np.nanmin(Yz[Yz>0]):.3g} .. {np.nanmax(Yz):.3g}")
        # write artifacts
        smp = []
        for i, (blk, subj, vis) in enumerate(parsed):
            ep = f"B{blk}P{subj}" if blk is not None else f"P{subj}"
            smp.append(dict(sample_id=ids[i], time=times[i], block=blk, subject=subj, visit=vis, episode=ep))
        with open(f"{OUT}/{study}_samples.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(smp[0].keys()))
            w.writeheader(); w.writerows(smp)
        with open(f"{OUT}/{study}_features.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        np.save(f"{OUT}/{study}_Y.npy", Y.astype(np.float32))
        json.dump(dict(study=study, n_feat=Y.shape[0], n_samp=Y.shape[1], times=dict(Counter(times)),
                       episodes=len(eps), missing=int(n_nan), zero_frac=frac_zero),
                  open(f"{OUT}/{study}_qc.json", "w"), indent=1)
    txt = "\n".join(report)
    print(txt)
    (OUT / "qc_parse.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    main(parser.parse_args().inputs)

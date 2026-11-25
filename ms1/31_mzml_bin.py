#!/usr/bin/env python3
"""mzML MS1 m/z binning (lxml + numpy).

Writes one float32 vector per file: sum of MS1 intensity in 0.01 Da bins, 200-1800 Da.
Usage: python3 31_mzml_bin.py <mzml_dir> <out_dir> [--workers N]
"""
import base64, os, tempfile, time, zlib
import numpy as np
from lxml import etree

NS = "{http://psi.hupo.org/ms/mzml}"
LO, HI, W = 200.0, 1800.0, 0.01
NB = int((HI - LO) / W)

ACC_MSLEVEL = "MS:1000511"
ACC_MZ = "MS:1000514"
ACC_INT = "MS:1000515"
ACC_F32 = "MS:1000521"
ACC_F64 = "MS:1000523"
ACC_ZLIB = "MS:1000574"


def _cvs(el):
    return {c.get("accession"): c.get("value") for c in el.iter(NS + "cvParam")}


def bin_one(path):
    acc = np.zeros(NB, dtype=np.float64)
    n_ms1 = 0
    for _, sp in etree.iterparse(path, events=("end",), tag=NS + "spectrum", huge_tree=True):
        cv = _cvs(sp)
        lvl = cv.get(ACC_MSLEVEL)
        if str(lvl) not in ("1", "1.0"):
            sp.clear()
            continue
        arrays = []
        for bda in sp.iter(NS + "binaryDataArray"):
            acv = _cvs(bda)
            kind = None
            if ACC_MZ in acv:
                kind = "mz"
            elif ACC_INT in acv:
                kind = "int"
            if kind is None:
                continue
            dt = np.float64 if ACC_F64 in acv else np.float32
            b = bda.find(NS + "binary")
            if b is None or not b.text:
                continue
            raw = base64.b64decode(b.text)
            if ACC_ZLIB in acv:
                raw = zlib.decompress(raw)
            arrays.append((kind, np.frombuffer(raw, dtype=dt)))
        d = dict(arrays)
        if "mz" in d and "int" in d:
            n_ms1 += 1
            mz, it = d["mz"], d["int"]
            if mz.size:
                idx = ((mz - LO) / W).astype(np.int64)
                ok = (idx >= 0) & (idx < NB)
                if ok.any():
                    acc += np.bincount(idx[ok], weights=it[ok], minlength=NB)
        sp.clear()
        while sp.getprevious() is not None:
            del sp.getparent()[0]
    return acc.astype(np.float32), n_ms1


def work(arg):
    path, out = arg
    stem = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(out, stem + ".npy")
    if os.path.exists(dst):
        return stem, -1, "skip"
    t = time.time()
    pending = None
    try:
        v, n = bin_one(path)
        with tempfile.NamedTemporaryFile(dir=out, prefix=f".{stem}.", suffix=".part",
                                         delete=False) as handle:
            pending = handle.name
            np.save(handle, v)
        os.replace(pending, dst)
        pending = None
        return stem, n, round(time.time() - t, 1)
    except Exception as e:
        return stem, -1, f"ERR {type(e).__name__}: {e}"
    finally:
        if pending is not None:
            os.unlink(pending)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mzml_dir", help="directory containing the study's mzML files")
    ap.add_argument("out_dir", help="directory for one .npy vector per input file")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker count (default: 1)")
    args = ap.parse_args()
    src, out, nproc = args.mzml_dir, args.out_dir, args.workers
    if nproc < 1:
        ap.error("--workers must be >= 1")
    os.makedirs(out, exist_ok=True)
    files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(".mzml"))
    if not files:
        ap.error(f"No mzML files in {src}")
    print(f"{len(files)} mzML, {nproc} workers, {NB} bins", flush=True)
    if nproc == 1:
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {work((f, out))}", flush=True)
        raise SystemExit
    from multiprocessing import Pool
    t0 = time.time()
    done = 0
    with Pool(nproc) as p:
        for stem, n, dt in p.imap_unordered(work, [(f, out) for f in files], chunksize=1):
            done += 1
            if done % 20 == 0 or (isinstance(dt, str) and dt.startswith("ERR")):
                el = time.time() - t0
                print(f"[{done}/{len(files)}] {stem} ms1={n} {dt}s elapsed={el:.0f}s "
                      f"eta={el/done*(len(files)-done)/60:.1f}min", flush=True)
    print(f"DONE {done} in {time.time()-t0:.0f}s", flush=True)

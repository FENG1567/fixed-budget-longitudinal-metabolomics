"""Fetch and checksum the deposited ST00389x matrices and MTBLS89 files."""

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from workbench_table import table_url


METABOLIGHTS = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS89/"
MATRICES = (
    ("ST003894", "MS", "AN006397", "83020caf3e5421278223f6399074af6d6a4acf284463707faad04d65a36bfe20"),
    ("ST003895", "MS", "AN006398", "fd252e57d7d8edcafbf0c9d05dd89a9195db15d52cada70706db690388f740e0"),
    ("ST003896", "Other", "AN006399", "b5995f06926cb4115250fa9dd08495096a3012a8042ba6b3b2bc2e3482968d7f"),
)
HUMET = (
    ("s_MTBLS89.txt", "2db4fe0cbc7d62f429782bfc6d47f71d22781fb5d00fce4bd249bf19f9b5502c"),
    ("m_MTBLS89_metabolite_profiling_mass_spectrometry_v2_maf.tsv",
     "1d9fe923a7ce0487fb2c3a1e1258bd55345290d6f34ec3f1ae5f49ef4141f6c8"),
    ("i_Investigation.txt", "9ea83ad8f3c5e1801746bea7da6d2c03e6abc344052baf3c77f15ad8c11dd2bc"),
)


def verify(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, received {actual}")


def fetch(path, url, expected):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".part")
        request = Request(url(), headers={"User-Agent": "Longitudinal-omics-reproduction/1.0"})
        try:
            with urlopen(request, timeout=90) as response, pending.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            verify(pending, expected)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    verify(path, expected)
    print(f"Verified {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    args = parser.parse_args()
    raw, humet = (args.inputs / name for name in ("01_data/raw", "01_data/external/mtbls89"))
    for study, kind, analysis, sha in MATRICES:
        fetch(raw / f"{study}_data.txt", lambda: table_url(study, kind, analysis), sha)
    for name, sha in HUMET:
        fetch(humet / name, lambda: METABOLIGHTS + name, sha)


if __name__ == "__main__":
    main()

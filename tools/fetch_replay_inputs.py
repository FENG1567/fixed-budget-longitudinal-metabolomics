#!/usr/bin/env python3
"""Fetch the public human metabolomics inputs used by the replay analyses.

The default command fetches the two PoMet DataverseNO matrices and the
ST003348 Workbench quantitative matrix and sample factors:

    python tools/fetch_replay_inputs.py --inputs inputs

Add the exploratory ST001251 table and the female/male Zenodo
workbooks with:

    python tools/fetch_replay_inputs.py --inputs inputs --include-exploratory

Existing files are checked first; ``--force`` replaces them only after a
successful download and checksum check. The script uses only Python's standard
library and preserves the repository-specific input layout.

Source endpoints:

* DataverseNO PoMet files 205866 and 196764.
* Metabolomics Workbench ST003348 ``data`` and ``factors`` REST endpoints.
* Exploratory ST001251 downloaded as the Workbench tab-delimited study table.
* Zenodo record 20699015 files ``metabolite_data_female.csv`` and
  ``metabolite_data_male.csv``. These exact record names were resolved from
  the public record metadata. The downloaded bytes are workbook containers
  despite the record's .csv suffix; this script preserves the original bytes
  as female.csv and male.csv.

Archived-byte SHA-256 values are enforced for all seven source files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen

from workbench_table import table_url


SOURCES = {
    "pomet_205866.csv": (
        "https://dataverse.no/api/access/datafile/205866",
        "77127e4c3fee223fb0f61caaada2593d0ff4fba48d69949048d9f9347a3dfc08",
    ),
    "pomet_196764.csv": (
        "https://dataverse.no/api/access/datafile/196764",
        "17a4e13f368233322c7bb690fdd92445553ab17e5ce56e6a035f38b9b332eafc",
    ),
    "ST003348_quant.json": (
        "https://www.metabolomicsworkbench.org/rest/study/study_id/ST003348/data",
        "f6821b1eecf7f77cee1dd41710b548dc1f3679ced8a1e8ceb271835ec221e55d",
    ),
    "st003348_factors.json": (
        "https://www.metabolomicsworkbench.org/rest/study/study_id/ST003348/factors",
        "e31bb851873fee1d185a2d91d8b5553637145d799793d416ecf2be6c3f8c3329",
    ),
}

EXPLORATORY_SOURCES = {
    "01_data/external/st001251/data.txt": (
        None,
        "8bd0749be07f14c7012c72314d75cb4b6ff9a0e0eb279cfe5aff22d7fa7311af",
    ),
    "zenodo/female.csv": (
        "https://zenodo.org/api/records/20699015/files/metabolite_data_female.csv/content",
        "21a7360814047a95c83b6bdc61d0905e1d936e291b6b596b6a9a4d28cacb71de",
    ),
    "zenodo/male.csv": (
        "https://zenodo.org/api/records/20699015/files/metabolite_data_male.csv/content",
        "fee694300649a525b75182fb8ba82a6f121941e68d6dda2a5f889dece4d3be27",
    ),
}

USER_AGENT = "public-metabolomics-replay-input-fetcher/1.0"
BLOCK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_one(name: str, url: str | None, expected_sha256: str,
              output_dir: Path, force: bool) -> str:
    destination = output_dir / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Existing file checksum mismatch for {destination}: "
                f"{actual} != {expected_sha256}; pass --force to replace it"
            )
        print(f"verified: {destination} sha256={actual}")
        return actual

    temporary_path: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    if url is None:
        url = table_url("ST001251", "MS", "AN002077")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=90) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    block = response.read(BLOCK_SIZE)
                    if not block:
                        break
                    temporary.write(block)
                    digest.update(block)
                    byte_count += len(block)
        actual = digest.hexdigest()
        if byte_count == 0:
            raise RuntimeError(f"Empty response from {url}")
        if actual != expected_sha256:
            raise RuntimeError(
                f"Downloaded checksum mismatch for {name}: "
                f"{actual} != {expected_sha256} ({url})"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"downloaded: {destination} bytes={byte_count} "
              f"sha256={actual} (verified)")
        return actual
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Could not fetch {name} from {url}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=Path("inputs"),
        help="root directory for downloaded inputs (default: ./inputs)",
    )
    parser.add_argument(
        "--include-exploratory", action="store_true",
        help="also fetch ST001251 and the Zenodo female/male workbooks",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace existing files after the new download passes validation",
    )
    args = parser.parse_args(argv)
    output_dir = args.inputs.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {f"replay/{name}": entry for name, entry in SOURCES.items()}
    if args.include_exploratory:
        sources.update(EXPLORATORY_SOURCES)

    try:
        for name, (url, expected_sha256) in sources.items():
            fetch_one(name, url, expected_sha256, output_dir, args.force)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

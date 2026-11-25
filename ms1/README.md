# Supplementary S3 raw-signal checks

This directory contains the reproducible code path for the exploratory MS1
mass-window checks described in Supplementary Information S3. The workflow is
limited to publicly deposited human metabolomics data from ST003894 and
ST003895. No participant-level or raw mass-spectrometry data are included here.

## Provenance and scope

The SHA-256 hashes below identify the original analysis files by basename.
The copies here preserve the original binning, filtering, transformations,
correlation, and target-window calculations. Changes are limited to command-line arguments, input/output
paths, and a single-process default for the binner so that the scripts can be
run outside the original fixed directory tree.

| Release script | Original file | Original SHA-256 |
|---|---|---|
| `31_mzml_bin.py` | `31_mzml_bin.py` | `4f9137faa7e1bbf0faa352739c62a0165a395df89fc8abb1abe3aed0cea03dd7` |
| `43_reliability2.py` | `43_reliability2.py` | `05211f2d7a647ae2535e3bb6ea25b9e5f77cf1fde9c1ff7e1dbcf5e25f5e0352` |
| `45_rho_norm.py` | `45_rho_norm.py` | `af0cc5f4da719f30942f9203a12a8b9694582d689e2f43e7d0cec34f8eff1aa2` |
| `49_raw_summary.py` | `49_raw_summary.py` | `0b09b005f3773a2ef71e5ed85593afb0af98ce33be31757eca0c1528bff5c817` |
| `51_targeted.py` | `51_targeted.py` | `f0936334ae1e58e9bb618317b0116c52fea8ad4f7f1eefab974db3ca63a63cf7` |

The S3 headline values come from distinct original checks:

* `49_raw_summary.py`, default width 10: source spectra are first stored in
  0.01-Da bins over m/z 200–1800; the reported correlation summary sums each
  run of 10 adjacent bins (0.10 Da) and calculates median squared correlation
  of 120- and 180-minute log₂ changes across the common complete episodes.
  This reproduces the approximate values 0.292 (ST003894) and 0.347
  (ST003895) in S3. The 0.01-Da source vectors and 0.10-Da summary bins are
  distinct stages.
* `43_reliability2.py`: a separate bin-width sensitivity over w = 1, 5, 10,
  and 50 source bins (0.01–0.50 Da). It reports each width explicitly; at
  w=1 the ST003894 result is 0.268, so that number should not be confused with
  the S3 0.292 summary value. This sensitivity script requires the same
  complete episode set at every time; use it with ST003894, not the unbalanced
  ST003895 bins.
* `45_rho_norm.py`, width 10: repeat the 120-versus-180-minute feature/bin
  correlation after raw, total-ion-current, median, and probabilistic-quotient
  normalization. S3 reports the TIC, median, and PQN values for ST003894.
* `51_targeted.py`: evaluate the eight listed m/z windows (±0.02 Da) in
  ST003894. This is a mass-window search, not chromatographic peak finding or
  structural identification. The S3 maximum squared correlation is 0.706.
* `49_raw_summary.py` also reports profile reliability, normalization
  variants, and intensity-decile summaries. `45_rho_norm.py` is the dedicated
  normalization check and independently reproduces the S3 ST003894
  normalization values.

Signals are binned by m/z only. These checks do not perform retention-time
alignment, chromatographic peak identification, isotope grouping, adduct
grouping, or metabolite identification. A selected window may contain signal
from multiple chemical species. The reported correlations describe these
signal representations, not analyte-level assay measurements.

## Inputs

Install Python 3 and the two packages in `requirements.txt`. Obtain the study
tables and generate sample maps with the release parser:

```bash
python tools/parse_profile_matrices.py --inputs inputs
```

The parser writes `inputs/01_data/parsed/ST003894_samples.csv` and
`ST003895_samples.csv` for this workflow.

The public MS1 archives are the original Metabolomics Workbench downloads:

* [ST003894 mzML archive](https://www.metabolomicsworkbench.org/studydownload/ST003894_mzml.zip)
* [ST003895 mzML archive](https://www.metabolomicsworkbench.org/studydownload/ST003895_mzml.zip)

The abundance tables used to generate sample maps are also available directly:

* [ST003894 data table](https://www.metabolomicsworkbench.org/data/data_table_output.php?STUDY_ID=ST003894&STUDY_TYPE=MS&RESULT_TYPE=1)
* [ST003895 data table](https://www.metabolomicsworkbench.org/data/data_table_output.php?STUDY_ID=ST003895&STUDY_TYPE=MS&RESULT_TYPE=1)

Extract each mzML archive directly into its study directory, such as
`inputs/01_data/mzml/894/` and `inputs/01_data/mzml/895/`. Preserve archive
file basenames and case (for example, `<sample_id>.mzML`). The binner
accepts the `.mzML` suffix case-insensitively, while sample-map matching uses
the bin filename stem exactly. The ST003895 archive includes a known
truncated XML file; it reports a parse error and produces no bin vector. The
other files can be processed normally.

The raw-signal scripts use:

* `<bins_dir>/*.npy`: one vector per source-file basename, with 160,000
  float32 values for 0.01-Da bins over m/z 200–1800. These are produced from
  the extracted mzML archives by `31_mzml_bin.py`.
* `<samples_csv>`: the matching study map, with `sample_id`, `episode`, and
  `time` fields. The `sample_id` must match the bin-vector filename stem.


## Run

Run the commands from the release root. Prepare sample maps first. The two
binning commands scan entire study directories and should run on a suitable
server. If existing bins are available, skip the binning commands.

```bash
python -m pip install -r ms1/requirements.txt

# 1. Build sample maps from the public profile matrices.
python tools/fetch_profile_inputs.py --inputs inputs
python tools/parse_profile_matrices.py --inputs inputs

# 2. Optional: bin both mzML archives.
python ms1/31_mzml_bin.py inputs/01_data/mzml/894 inputs/01_data/bins/894
python ms1/31_mzml_bin.py inputs/01_data/mzml/895 inputs/01_data/bins/895

# 3. Headline summaries, normalization sensitivity, and target windows.
bash ms1/run_s3_checks.sh inputs/01_data/bins inputs/01_data/parsed build/ms1
python verify.py --stage ms1 --rerun build/ms1

# 4. Dedicated normalization and targeted-window commands, if run separately.
python ms1/45_rho_norm.py 894 --bins inputs/01_data/bins/894 \
  --samples inputs/01_data/parsed/ST003894_samples.csv
python ms1/51_targeted.py 894 --bins inputs/01_data/bins/894 \
  --samples inputs/01_data/parsed/ST003894_samples.csv

# 5. Optional bin-width sensitivity analysis.
python ms1/43_reliability2.py 894 --bins inputs/01_data/bins/894 \
  --samples inputs/01_data/parsed/ST003894_samples.csv \
  --output build/ms1/raw_reliability2_894.json

# 6. Optional 0.1-Da companion summary (already run by the entry point).
python ms1/49_raw_summary.py 894 --bins inputs/01_data/bins/894 \
  --samples inputs/01_data/parsed/ST003894_samples.csv \
  --output build/ms1/raw_summary_894.json
```

For large-scale binning, submit the command as a background job on the data
server and retain its PID and log path. The default binner is serial;
`--workers > 1` enables multiprocessing. Completed vectors are written atomically
and are skipped in a resumed run. The analysis commands load all available bin
vectors into memory. The verification command compares the two archived summary
JSON files and the dedicated normalization and target-window output captured
by the shell runner.

To smoke-test the binner without processing a full archive, select one already
extracted mzML file and run it through a one-file directory:

```bash
mkdir -p build/ms1-smoke/input build/ms1-smoke/bins
sample_file=$(find inputs/01_data/mzml/894 -maxdepth 1 -type f -iname '*.mzml' -print -quit)
test -n "$sample_file"
sample_stem=$(basename "$sample_file")
sample_stem=${sample_stem%.*}
ln -s "$PWD/$sample_file" build/ms1-smoke/input/
python ms1/31_mzml_bin.py build/ms1-smoke/input build/ms1-smoke/bins --workers 1
python -c "import numpy as np; x=np.load('build/ms1-smoke/bins/${sample_stem}.npy'); assert x.shape == (160000,) and np.isfinite(x).all() and (x >= 0).all(); print('one-file bin vector: PASS')"
```

Run even this smoke check on the data server when the mzML file is large. It
uses the existing archive extraction and does not download or copy raw data.

## Scope and limits

Counts in the portable scripts refer to the inputs supplied at runtime. For
ST003895, the extraction yields 479 mzML files, one of which is malformed and
excluded, leaving 478 bin vectors; the paired analyses use the 93 subjects with
complete trajectories. The binner was checked on a single-MS1 fixture, and the
ST003894 bin-width result was checked on archived bin vectors. The eight
targeted windows cover abundant lipid classes and are defined in
`51_targeted.py`.

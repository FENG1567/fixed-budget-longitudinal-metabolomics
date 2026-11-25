# Supplement S3 exploratory panel checks

`reproduce_panels.py` is a portable extraction of the archived producers
`B2_panel_k.py` (using the fold helpers in `B0_panel_local.py`),
`B4_stability.py`, and `B6_loso.py`. Its reported-check path preserves their
feature filter (`Y.min(axis=1) > 0`), log2(x + 1)
baseline-change transformation, top-20 ranking by absolute Pearson correlation,
fold/split seeds, and cross-study common-feature restriction. The only changes
are input/output paths and limiting the half-sample calculation to the
ST003894 result reported in Supplement S3.

## Inputs

The private parsed-input directory must contain these files:

```text
ST003894_Y.npy       ST003894_features.txt       ST003894_samples.csv
ST003895_Y.npy       ST003895_features.txt       ST003895_samples.csv
ST003896_Y.npy       ST003896_features.txt       ST003896_samples.csv
```

The sample CSV columns are `sample_id`, `time`, `episode`, and `subject`.
No raw or parsed study data are included in this release directory. Python 3,
NumPy, and the standard library are required.

## Run

From the release root, choose a new output directory outside the input
directory:

```sh
python panels/reproduce_panels.py \
  --parsed inputs/01_data/parsed \
  --out build/panels
```

The output directory must not already exist. The script writes
`panel_results.csv` and `panel_results.json` only; it does not copy inputs.

## Expected comparison

| Check | Archived value | Output key |
|---|---:|---|
| Five subject-grouped folds, ST003894 top-20 held-out squared correlation | 0.720 | `five_fold_ST003894_top20_rho2` |
| Forty half-sample splits, ST003894 mean held-out squared correlation | 0.604 | `half_split_ST003894_top20_mean_rho2` |
| Forty half-sample splits, mean top-20 Jaccard similarity | 0.26 | `half_split_ST003894_top20_mean_jaccard` |
| ST003895 panel to ST003896 evaluation, k=20 | 0.299 (archived 0.2987) | `cross_study_ST003895_to_ST003896_top20_rho2` |
| ST003896 panel to ST003895 evaluation, k=20 | 0.437 | `cross_study_ST003896_to_ST003895_top20_rho2` |

The five-fold calculation preserves B2's seed-11 fold sequence: PC1, top-5,
top-10, then top-20. The top-5 and top-10 calls consume shuffles before the
reported top-20 call.
The half-sample result is the first profile in B4, so its seed-31 stream is
identical to the archived ST003894 result. Transfer eligibility is the
intersection of positive-complete feature names in both study matrices, as in
B6. The cross-study estimate is exploratory and not an independent external
validation.

# The cost of small pilots in fixed-budget sampling for longitudinal metabolomics studies

This repository accompanies the study by Shuo Feng, Chao Shi, Tianming Liu, Haoran Ding, Fenghe Hang, Zhenghao Huang and Liwei Liu. The analyses compare direct and nested two-phase sampling for estimating mean late log change under fixed budgets. The empirical replay uses finite source rosters and complete eligible panels. The simulations evaluate specified data-generating scenarios.

## Quick reproduction

Use CPython 3.12. On Windows, `py -3.12` can select this interpreter. Create and activate a virtual environment for your platform, then run these commands from the repository root.

```text
python -m pip install -r requirements.txt
python verify.py
python reproduce.py display
python verify.py --stage display --rerun build/display
```

`build/display/` contains Figures 1–4 in PDF, numeric source CSVs for Tables 1–2, and a re-enumerated 40-scenario finite-roster integer grid. Figure 1 is computed analytically. Figures 2–4 are regenerated from the published aggregate CSVs and the stored plot labels and settings. Display verification compares the numeric tables, grid and figure-input JSON. PDF timestamps and font encodings can differ by platform. To plot from reruns, copy the released `data/` folder to a working directory, overlay the rerun `profile_estimates.csv`, `summary_policy_cross.csv`, `availability_flow.csv` and `operational_policy_metrics.csv`, and select that directory with `--data`.

## Regenerate the statistics

The cost and simulation stages need only the Python environment. Each stage writes to its own new directory. Pass `--out` to select a separate directory for a subsequent run.

```text
python reproduce.py cost
python reproduce.py simulation
python verify.py --stage cost --rerun build/cost
python verify.py --stage simulation --rerun build/simulation
```

For empirical stages, retrieve public inputs into the ignored `inputs/` directory. The fetchers verify archived SHA-256 digests before use. [Data sources and exact input paths](DATA_SOURCES.md)

```text
python tools/fetch_replay_inputs.py --inputs inputs --include-exploratory
python tools/fetch_profile_inputs.py --inputs inputs
python tools/parse_profile_matrices.py --inputs inputs
python reproduce.py empirical --inputs inputs
python reproduce.py interleaved --inputs inputs
python reproduce.py profiles --inputs inputs
python reproduce.py multi-aux --inputs inputs
python reproduce.py long-visit --inputs inputs
python reproduce.py zenodo --inputs inputs
```

The empirical replay runs 400 partitions with 200 masks per partition and includes leave-one-participant and pilot-size checks. The `interleaved` stage reproduces the alternative partition schedule reported in Supplementary Section S4. Its `summary.csv` uses the original strategy names. Use `python verify.py --stage <stage> --rerun build/<stage>` for released numeric tables and the block-wise replay configuration. The comparator accepts floating-point rounding differences. For the operational simulation it aligns the published column order, whose export omits two redundant cost fields and shortens one prose field.

The supplementary feature-panel and raw-spectrum checks have their own input instructions. The corresponding verification stage names are `panels` and `ms1`. [Feature-panel checks](panels/README.md) · [MS1 mass-window checks](ms1/README.md)

## Files and licences

| Path | Contents |
|---|---|
| `tools/` | Statistical analysis, figure and table builders, and public-source retrieval and preprocessing |
| `data/` | Author-produced aggregate outputs, figure settings and run protocol |
| `panels/`, `ms1/` | Supplementary analysis code and input instructions |
| `reproduce.py`, `verify.py` | Run individual stages, verify release integrity and compare reruns |
| `requirements.txt`, `CITATION.cff`, `SHA256SUMS.txt` | Software versions, citation metadata and release checksums |

Author-written software and its documentation are available under [MIT](LICENSE). The authors' aggregate results in `data/` are available under [CC BY 4.0](data/LICENSE). Cite the authors and identify any changes when reusing these results. Data obtained from the source repositories retain the applicable provider terms. [Output and variable guide](data/README.md) · [Data sources](DATA_SOURCES.md)

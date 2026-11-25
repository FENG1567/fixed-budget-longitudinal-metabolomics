"""Check release checksums or compare a rerun against distributed numeric results."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "cost": ("cost_scenarios.csv", "cost_price_sensitivity.csv", "cost_scenario_summary.csv"),
    "display": ("empirical_integer_design_grid.csv", "figure_inputs.json",
                "table1_profiles.csv", "table2_policies.csv"),
    "simulation": ("operational_policy_metrics.csv", "operational_design_grid.csv"),
    "empirical": ("summary_policy_cross.csv", "split_policy_cross.csv", "availability_flow.csv",
                  "feature_availability.csv", "lopo_policy_sensitivity.csv", "m_sensitivity_summary.csv",
                  "m_sensitivity_split.csv", "pomet_feature_sensitivity.csv", "empirical_protocol.json"),
    "interleaved": ("interleaved_summary.csv",),
    "profiles": ("profile_estimates.csv", "paired_cluster_fdr_summary.csv"),
    "long-visit": ("long_visit_replay.csv",),
    "zenodo": ("zenodo_lunch_replay.csv",),
    "multi-aux": ("multi_aux_metrics.json",),
    "panels": ("s3_panel_checks.csv",),
    "ms1": ("raw_summary_894.json", "raw_summary_895.json"),
}
RUN_FILENAMES = {"interleaved_summary.csv": "summary.csv", "s3_panel_checks.csv": "panel_results.csv"}


def verify_manifest():
    entries = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for entry in entries:
        expected, name = entry.split("  ", 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Unsafe manifest path: {name}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Verified {len(entries)} SHA-256 entries.")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        return list(csv.reader(handle))


def equal(a, b, location):
    if a == b:
        return
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        for key in a:
            equal(a[key], b[key], f"{location}/{key}")
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (left, right) in enumerate(zip(a, b, strict=True), 1):
            equal(left, right, f"{location}/{index}")
        return
    if isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
        try:
            x, y = float(a), float(b)
        except ValueError:
            pass
        else:
            if (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-11):
                return
    raise ValueError(f"Mismatch at {location}: {str(a)[:100]!r} vs {str(b)[:100]!r}")


def simulation_csv(a, b, name):
    # The manuscript CSV uses the same results with the cost columns rearranged.
    columns, actual_columns = a[0], b[0]
    extra = {"pilot_cost_if_pilot_dependent_policy", "remaining_main_budget_after_pilot"}
    if not set(columns) <= set(actual_columns) or set(actual_columns) - set(columns) - extra:
        raise ValueError(f"Unexpected schema change: {name}")
    keep = [column for column in columns if column != "pilot_and_main"]
    expected = [[row[columns.index(column)] for column in keep] for row in a[1:]]
    actual = [[row[actual_columns.index(column)] for column in keep] for row in b[1:]]
    equal(expected, actual, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=GROUPS)
    parser.add_argument("--rerun", type=Path)
    args = parser.parse_args()
    if bool(args.stage) != bool(args.rerun):
        parser.error("Provide --stage and --rerun together.")
    if not args.stage:
        verify_manifest()
        return
    for name in GROUPS[args.stage]:
        if name == "empirical_protocol.json":
            expected = read(ROOT / "data" / name)
            actual = read(args.rerun / "protocol.json")
            keys = ("source_hashes_sha256", "seed", "outer_splits", "nested_random_masks_per_split",
                    "pilot_m", "g_values", "delta_v_grid", "individual_data_exported", "structures")
            for key in keys:
                equal(expected[key], actual[key], f"{name}/{key}")
            print(f"Matched {name} configuration")
        else:
            expected = read(ROOT / "data" / name)
            actual = read(args.rerun / RUN_FILENAMES.get(name, name))
            if args.stage == "simulation":
                simulation_csv(expected, actual, name)
            else:
                equal(expected, actual, name)
            print(f"Matched {name}")
    if args.stage == "ms1":
        norm = (args.rerun / "normalization_894.txt").read_text(encoding="utf-8")
        targeted = (args.rerun / "targeted_windows_894.txt").read_text(encoding="utf-8")
        medians = dict(re.findall(r"(?m)^\s*(\S+)\s+rho2_median=([\d.]+)", norm))
        equal({"tic": "0.312", "median": "0.298", "pqn": "0.299"},
              {name: medians[name] for name in ("tic", "median", "pqn")}, "MS1 normalization")
        windows = [float(value) for value in re.findall(r"\brho2=([\d.]+)", targeted)]
        equal(8, len(windows), "MS1 targeted windows")
        equal(0.706, max(windows), "MS1 targeted window maximum")
        print("Matched S3 normalization medians and the target-window maximum.")


if __name__ == "__main__":
    main()

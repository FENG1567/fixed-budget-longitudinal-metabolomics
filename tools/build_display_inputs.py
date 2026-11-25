#!/usr/bin/env python3
"""Rebuild main-figure inputs and compact main-table CSVs from aggregate data.

Example:
    python build_display_inputs.py --data path/to/03_Supplementary/data \
        --out path/to/release/display_inputs

All estimates and errors come from the aggregate CSVs. The archived
figure_inputs.json supplies display labels and plotting settings only; its
numeric payload is checked against the rebuilt values and never used as data.
Only the standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PROFILE_SPECS = (
    ("CN-lipid", "CN-lipid (ST003894)", "visit unit"),
    ("CN-metab", "CN-metab (ST003895)", "visit unit"),
    ("CN-meal", "CN-meal (ST003896)", "visit unit"),
    ("DE-OGTT", "HuMet OGTT", "participant;"),
    ("DE-OLTT", "HuMet OLTT", "participant;"),
)

LOSS_COLUMNS = (
    {
        "Direct after pilot": "matched_gold_mse_mean",
        "Pilot allocation + pilot slope": "pilot_alloc_pilot_slope_mse_mean",
        "Pilot allocation + reference slope": "pilot_alloc_oracle_slope_mse_mean",
        "Reference allocation + pilot slope": "oracle_alloc_pilot_slope_mse_mean",
        "Reference allocation + slope": "oracle_alloc_oracle_slope_mse_mean",
    },
    {
        "Direct without pilot": "same_roster_direct_mse_mean_mean",
        "Raw pilot rule": "same_roster_policy_mse_mean_mean",
        "Adjusted pilot rule": "same_roster_adjusted_r2_policy_mse_mean_mean",
        "Adjusted rule + shrink": "same_roster_adjusted_r2_shrink_policy_mse_mean_mean",
    },
)

STRESS_POLICIES = {
    "Direct without pilot": "direct_no_pilot",
    "Direct after pilot": "direct_matched_pilot",
    "Oracle": "oracle",
    "Raw pilot rule": "operational_rawR2_raw_slope",
    "Adjusted pilot rule": "operational_adjustedR2_raw_slope",
}

TABLE2_FIELDS = (
    "cohort",
    "dataset",
    "delta_v",
    "g",
    "outer_splits",
    "nested_decisions",
    "pilot_borrow_fraction",
    "direct_after_pilot_mse",
    "raw_pilot_mse",
    "raw_minus_direct_mse",
    "adjusted_rule_mse",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric field {key!r}; row={row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Expected finite field {key!r}; row={row}")
    return value


def require_columns(rows: list[dict[str, str]], path: Path, columns: set[str]) -> None:
    if not rows:
        raise ValueError(f"No data rows in {path}")
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def find_one(rows: list[dict[str, str]], label: str, path: Path) -> dict[str, str]:
    matches = [row for row in rows if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"Expected one archived {label!r} record in {path}; found {len(matches)}")
    return matches[0]


def check_value(source: float, archived: Any, context: str, warnings: list[str]) -> None:
    try:
        old = float(archived)
    except (TypeError, ValueError):
        warnings.append(f"Archived value missing or nonnumeric for {context}; CSV value used")
        return
    if not math.isclose(source, old, rel_tol=0.0, abs_tol=1e-12):
        warnings.append(f"Archived value differs for {context}; CSV value used")


def build_profiles(
    source_rows: list[dict[str, str]], archived: dict[str, Any], warnings: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require_columns(
        source_rows,
        Path("profile_estimates.csv"),
        {
            "profile", "source", "challenge", "early_min", "late_min", "target_unit",
            "transform", "n_subjects", "n_visit_units", "features_source_total",
            "features_retained", "median_rho2", "cluster_bootstrap_ci95_low",
            "cluster_bootstrap_ci95_high",
        },
    )
    archived_profiles = archived.get("profiles", [])
    table_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for profile, label, unit_fragment in PROFILE_SPECS:
        matches = [
            row for row in source_rows
            if row["profile"] == profile
            and row["transform"] == "log2(x+1)"
            and unit_fragment in row["target_unit"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one primary profile row for {profile}; found {len(matches)}")
        row = matches[0]
        estimate = number(row, "median_rho2")
        low = number(row, "cluster_bootstrap_ci95_low")
        high = number(row, "cluster_bootstrap_ci95_high")
        archived_match = next((item for item in archived_profiles if item.get("label") == label), None)
        if archived_match is None:
            warnings.append(f"No archived display record for {label}; source mapping used")
        else:
            check_value(estimate, archived_match.get("estimate"), f"Figure 2 {label} estimate", warnings)
            check_value(low, archived_match.get("lo"), f"Figure 2 {label} lower interval", warnings)
            check_value(high, archived_match.get("hi"), f"Figure 2 {label} upper interval", warnings)
        figure_rows.append({"label": label, "estimate": estimate, "lo": low, "hi": high})
        table_rows.append(
            {
                "profile": label,
                "source": row["source"],
                "challenge": row["challenge"],
                "participants": row["n_subjects"],
                "visit_units": row["n_visit_units"],
                "early_min": row["early_min"],
                "late_min": row["late_min"],
                "features_retained": row["features_retained"],
                "features_source_total": row["features_source_total"],
                "transform": row["transform"],
                "analysis_unit": row["target_unit"],
                "median_rho2": row["median_rho2"],
                "ci95_low": row["cluster_bootstrap_ci95_low"],
                "ci95_high": row["cluster_bootstrap_ci95_high"],
            }
        )
    if len(archived_profiles) != len(figure_rows):
        warnings.append(
            f"Archived Figure 2 has {len(archived_profiles)} records; rebuilt {len(figure_rows)} profiles"
        )
    return figure_rows, table_rows


def build_losses(
    summary_rows: list[dict[str, str]], archived: dict[str, Any], warnings: list[str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    source_key = ("PoMet_205866_primary", 8.0, 12)
    matches = [
        row for row in summary_rows
        if row["dataset"] == source_key[0]
        and number(row, "delta_v") == source_key[1]
        and number(row, "g") == source_key[2]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one primary PoMet delta_v=8, g=12 summary; found {len(matches)}")
    source = matches[0]
    panels = archived.get("losses", [])
    if len(panels) != len(LOSS_COLUMNS):
        raise ValueError(f"Expected {len(LOSS_COLUMNS)} archived Figure 3 panels; found {len(panels)}")
    rebuilt: list[dict[str, Any]] = []
    table_sources: dict[str, str] = {}
    for panel_index, (panel, columns) in enumerate(zip(panels, LOSS_COLUMNS)):
        panel_copy = dict(panel)
        values = []
        archived_values = panel.get("values", [])
        seen = set()
        for item in archived_values:
            label = item.get("label")
            if label not in columns:
                raise ValueError(f"Unmapped Figure 3 label in panel {panel_index + 1}: {label!r}")
            if label in seen:
                raise ValueError(f"Duplicate Figure 3 label in panel {panel_index + 1}: {label!r}")
            seen.add(label)
            column = columns[label]
            value = number(source, column)
            check_value(value, item.get("value"), f"Figure 3 panel {panel_index + 1} {label}", warnings)
            value_copy = dict(item)
            value_copy["value"] = value
            values.append(value_copy)
            if panel_index == 1:
                table_sources[label] = column
        if seen != set(columns):
            raise ValueError(f"Incomplete Figure 3 panel {panel_index + 1}: labels={sorted(seen)}")
        panel_copy["values"] = values
        rebuilt.append(panel_copy)
    return rebuilt, table_sources


def build_stress(
    metrics: list[dict[str, str]], archived: dict[str, Any], warnings: list[str]
) -> list[dict[str, Any]]:
    require_columns(
        metrics,
        Path("operational_policy_metrics.csv"),
        {"scenario", "delta_v", "pilot_n_independent", "rho2_observed_target", "policy", "MSE"},
    )
    rebuilt: list[dict[str, Any]] = []
    for config in archived.get("stress", []):
        try:
            delta_v = float(config["title"].rsplit("=", 1)[1].strip())
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(f"Cannot read visit cost from Figure 4 title: {config}") from exc
        selected = [
            row for row in metrics
            if row["scenario"] == "homogeneous"
            and number(row, "delta_v") == delta_v
            and number(row, "pilot_n_independent") == 6
        ]
        xs = sorted({number(row, "rho2_observed_target") for row in selected})
        expected_n = len(config.get("series", [{}])[0].get("x", [])) if config.get("series") else 0
        if len(xs) != expected_n or expected_n == 0:
            raise ValueError(
                f"Figure 4 cost={delta_v:g} expected {expected_n} x values; found {len(xs)}"
            )
        config_copy = dict(config)
        series_rows = []
        for series in config.get("series", []):
            label = series.get("label")
            policy = STRESS_POLICIES.get(label)
            if policy is None:
                raise ValueError(f"Unmapped Figure 4 series label: {label!r}")
            ys: list[float] = []
            for x in xs:
                cell = [
                    row for row in selected
                    if row["policy"] == policy and number(row, "rho2_observed_target") == x
                ]
                if len(cell) != 1:
                    raise ValueError(
                        f"Expected one Figure 4 row for cost={delta_v:g}, x={x:g}, "
                        f"policy={policy}; found {len(cell)}"
                    )
                ys.append(number(cell[0], "MSE"))
            for position, (x, y) in enumerate(zip(xs, ys)):
                archived_x = series.get("x", [])[position]
                archived_y = series.get("y", [])[position]
                check_value(x, archived_x, f"Figure 4 cost={delta_v:g} {label} x[{position}]", warnings)
                check_value(y, archived_y, f"Figure 4 cost={delta_v:g} {label} y[{position}]", warnings)
            series_copy = dict(series)
            series_copy["x"] = xs
            series_copy["y"] = ys
            series_rows.append(series_copy)
        config_copy["series"] = series_rows
        rebuilt.append(config_copy)
    if not rebuilt:
        raise ValueError("No archived Figure 4 stress panels")
    return rebuilt


def build_table2(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_datasets = {
        "PoMet": "PoMet_205866_primary",
        "ST003348": "ST003348_REST_STAT_REC3_secondary",
    }
    result = []
    for cohort, dataset in source_datasets.items():
        for delta_v in (2.0, 8.0):
            matches = [
                row for row in summary_rows
                if row["dataset"] == dataset
                and number(row, "delta_v") == delta_v
                and number(row, "g") == 12
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one Table 2 row for {dataset}, delta_v={delta_v:g}, g=12; "
                    f"found {len(matches)}"
                )
            row = matches[0]
            fraction = number(row, "pilot_borrow_fraction")
            splits = number(row, "outer_splits")
            decisions_float = fraction * splits
            decisions = round(decisions_float)
            if not math.isclose(decisions_float, decisions, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"Borrow fraction does not imply an integer count: {row}")
            result.append(
                {
                    "cohort": cohort,
                    "dataset": dataset,
                    "delta_v": row["delta_v"],
                    "g": row["g"],
                    "outer_splits": row["outer_splits"],
                    "nested_decisions": f"{decisions}/{row['outer_splits']}",
                    "pilot_borrow_fraction": row["pilot_borrow_fraction"],
                    "direct_after_pilot_mse": row["gold_mask_mse_mean_mean"],
                    "raw_pilot_mse": row["policy_mask_mse_mean_mean"],
                    "raw_minus_direct_mse": row["paired_policy_minus_gold_mse_mean_mean"],
                    "adjusted_rule_mse": row["adjusted_r2_mask_mse_mean_mean"],
                }
            )
    return result


def build_integer_grid(summary_rows, availability, pilot_size):
    """Enumerate the finite-roster budget boundary in Supplementary Section S1."""
    roster = {row["dataset"]: int(row["source_roster_subjects"]) for row in availability}
    result = []
    for row in summary_rows:
        source = row["dataset"]
        h, g, cost = roster[source] - pilot_size, int(row["g"]), float(row["delta_v"])
        budget = g * (2 + cost)
        designs = [(n, k) for n in range(4, h + 1) for k in range(4, n + 1)
                   if 2 * n + (1 + cost) * k <= budget + 1e-8]
        crossings = [((1 / k - 1 / g) / (1 / k - 1 / n), n, k)
                     for n, k in designs if n > g and k < n]
        root, n0, k0 = min(crossings, default=(math.inf, 0, 0))
        result.append({
            "dataset": source, "held_out_H": h, "delta_v": cost, "direct_g": g,
            "post_pilot_budget": budget, "feasible_nested_pairs": len(designs),
            "max_feasible_N": max(n for n, _ in designs), "max_feasible_k": max(k for _, k in designs),
            "strict_integer_boundary_rho2": f"{root:.9f}" if root < 1 else "NA",
            "boundary_N": n0 if root < 1 else "NA", "boundary_k": k0 if root < 1 else "NA",
            "raw_policy_median_N": row["pilot_n_split_p50"], "raw_policy_median_k": row["pilot_k_split_p50"],
            "raw_policy_two_phase_fraction": row["pilot_borrow_fraction"],
            "raw_policy_mask_mse": f"{number(row, 'policy_mask_mse_mean_mean'):.9f}",
            "matched_direct_mask_mse": f"{number(row, 'gold_mask_mse_mean_mean'):.9f}",
        })
    return sorted(result, key=lambda row: (row["dataset"], row["delta_v"], -row["direct_g"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Aggregate CSV and archived JSON directory")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    data = args.data
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "profiles": data / "profile_estimates.csv",
        "summary": data / "summary_policy_cross.csv",
        "metrics": data / "operational_policy_metrics.csv",
        "archive": data / "figure_inputs.json",
        "availability": data / "availability_flow.csv",
        "protocol": data / "empirical_protocol.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input file(s) not found: " + ", ".join(missing))

    archived = json.loads(paths["archive"].read_text(encoding="utf-8-sig"))
    profile_source = read_csv(paths["profiles"])
    summary_source = read_csv(paths["summary"])
    stress_source = read_csv(paths["metrics"])
    warnings: list[str] = []

    figure_profiles, table1 = build_profiles(profile_source, archived, warnings)
    figure_losses, _ = build_losses(summary_source, archived, warnings)
    figure_stress = build_stress(stress_source, archived, warnings)
    figure_payload = {
        "stress": figure_stress,
        "profiles": figure_profiles,
        "losses": figure_losses,
    }

    write_csv(
        out / "table1_profiles.csv",
        table1,
        (
            "profile", "source", "challenge", "participants", "visit_units", "early_min", "late_min",
            "features_retained", "features_source_total", "transform", "analysis_unit", "median_rho2",
            "ci95_low", "ci95_high",
        ),
    )
    write_csv(out / "table2_policies.csv", build_table2(summary_source), TABLE2_FIELDS)
    (out / "figure_inputs.json").write_text(
        json.dumps(figure_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8-sig"))
    grid = build_integer_grid(summary_source, read_csv(paths["availability"]), protocol["pilot_m"])
    write_csv(out / "empirical_integer_design_grid.csv", grid, tuple(grid[0]))
    (out / "coverage.md").write_text(
        "# Display-input coverage\n\n"
        "Rebuilt Figure 2–4 numeric inputs from `profile_estimates.csv`, "
        "`summary_policy_cross.csv`, and `operational_policy_metrics.csv`. "
        "The archived `figure_inputs.json` supplies labels and plot settings only; "
        "its numeric values are taken from the aggregate CSVs.\n\n"
        "Generated `table1_profiles.csv` and `table2_policies.csv` at source precision. "
        "Enumerated `empirical_integer_design_grid.csv` from the published design and cohort sizes.\n\n"
        "Deliberate gaps: Figure 1's analytic boundary is drawn from the design model; "
        "supplementary cost tables, narrative manuscript values, and analyses requiring "
        "participant-level source data are outside this display stage. Table 2 is limited to the two primary "
        "cohorts and the published post-pilot direct sample size of 12.\n",
        encoding="utf-8",
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote Figure 2–4 inputs and main table CSVs to {out}")
    print("Figure 1 analytic values and supplementary cost/narrative tables are outside this utility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

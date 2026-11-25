"""Reproduce the illustrative price grids reported in Supplementary Section S2."""

import argparse
import csv
from math import erfc, pi, sqrt
from pathlib import Path

import numpy as np


N_DRAWS = 100_000
RANGES = {
    "USD": ([(45.5, 78), (20, 50), (25, 60), (5, 20)], (88, 225),
            [50, 65, 88, 100, 125, 150, 175, 225, 300, 400, 600, 800]),
    "CNY": ([(50, 120), (30, 80), (50, 150), (5, 20)], (400, 1000),
            [200, 250, 300, 400, 500, 650, 800, 1000, 1250, 1500, 2000]),
}


def logit(probability):
    return np.log(probability) - np.log1p(-probability)


def evaluate_cost(delta_v):
    c = 1 + delta_v
    cut = (3 * c + 2 + np.sqrt(8 * c * (c + 1))) / (c + 2) ** 2
    # Historical calibration retained solely to reproduce two archival columns.
    mean = logit(0.38977431423116715)
    pi_scale = (logit(0.7341493526050075) - logit(0.1287227063599749)) / (2 * 3.182446305284263)
    x = (logit(cut) - mean) / pi_scale
    tail_t = .5 - (np.arctan(x / sqrt(3)) + x * sqrt(3) / (x * x + 3)) / pi
    z = (logit(cut) - mean) / sqrt(2 * 0.16348452817605869)
    tail_normal = .5 * np.fromiter((erfc(float(value)) for value in z), dtype=float, count=len(z))
    return {
        "single_feature_0.29": float(np.mean(.29 > cut)),
        "conservative_panel_0.55": float(np.mean(.55 > cut)),
        "optimistic_panel_0.65": float(np.mean(.65 > cut)),
        "high_predictability_0.82": float(np.mean(.82 > cut)),
        "legacy_normal_predictive": float(np.mean(tail_normal)),
        "legacy_t_pi_sensitivity": float(np.mean(tail_t)),
    }


def calculate(output):
    rng = np.random.default_rng(20261122)
    rows = []
    for currency, (visit_ranges, assay_range, price_grid) in RANGES.items():
        visit = sum(rng.uniform(a, b, N_DRAWS) for a, b in visit_ranges)
        for price in [None, *price_grid]:
            delta_v = visit / (rng.uniform(*assay_range, N_DRAWS) if price is None else price)
            q1, median, q3 = map(float, np.quantile(delta_v, [.25, .50, .75]))
            rows.append({"currency": currency, "fixed_assay_price": price, "n_draws": N_DRAWS,
                         "dv_q1": q1, "dv_median": median, "dv_q3": q3, **evaluate_cost(delta_v)})

    output.mkdir(parents=True, exist_ok=True)
    with (output / "cost_price_sensitivity.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "cost_scenario_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        columns = [f"visit_component_{i}_{end}" for i in range(1, 5) for end in ("low", "high")]
        writer.writerow(["setting", *columns, "assay_price_low", "assay_price_high",
                         "median_delta_v", "q1_delta_v", "q3_delta_v"])
        for row in (r for r in rows if r["fixed_assay_price"] is None):
            code = row["currency"]
            label = {"USD": "United States", "CNY": "China"}[code]
            visit_ranges, assay_range, _ = RANGES[code]
            writer.writerow([label, *(cost for pair in visit_ranges for cost in pair), *assay_range,
                             row["dv_median"], row["dv_q1"], row["dv_q3"]])
        price50 = next(r for r in rows if r["currency"] == "USD" and r["fixed_assay_price"] == 50)
        writer.writerow([])
        writer.writerow(["illustrative_fixed_price_usd", "fixed_assay_price_usd", "fixed_rho2",
                         "n_visit_cost_draws", "threshold_crossing_fraction"])
        writer.writerow(["United States", 50, .82, N_DRAWS, price50["high_predictability_0.82"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    calculate(parser.parse_args().out)

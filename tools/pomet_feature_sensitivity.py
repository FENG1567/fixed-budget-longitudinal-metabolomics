"""Exact feature and scale sensitivity for the PoMet primary panel."""

import argparse
import csv
from pathlib import Path
import numpy as np

import frozen_benchmark as frozen


def exact_risk(x, y, beta, h, n, k):
    sigma2 = y.var(axis=0, ddof=1)
    if n == k:
        return sigma2 * (1 / n - 1 / h)
    residual2 = (y - beta * x).var(axis=0, ddof=1)
    return sigma2 * (1 / n - 1 / h) + residual2 * (1 / k - 1 / n)


def rule(q, h, delta_v, g):
    choice = frozen.predicted_allocation(float(q), h, delta_v, g)
    if choice["decision"] == "borrow":
        return choice["n_borrow"], choice["k_borrow"]
    return g, g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frozen.verify_sources()
    data = frozen.pomet_source("pomet_205866.csv", "PoMet_primary")
    x, y = data["X"], data["Y"]
    scale2 = y.var(axis=0, ddof=1)
    h, g, m = len(y) - 6, 12, 6
    rng = np.random.default_rng(20261122)
    splits = [rng.permutation(len(y)) for _ in range(400)]
    output = []
    for dv in (2.0, 8.0):
        losses = []
        for split in splits:
            pilot, held = split[:m], split[m:]
            xp, yp = x[pilot], y[pilot]
            xcp, ycp = xp - xp.mean(axis=0), yp - yp.mean(axis=0)
            r2, _ = frozen.correlation_squares(xp, yp)
            xx = (xcp * xcp).sum(axis=0)
            beta = np.divide((xcp * ycp).sum(axis=0), xx, out=np.zeros_like(xx), where=xx > 0)
            xh, yh = x[held], y[held]
            n, k = rule(r2.mean(), h, dv, g)
            sy2 = yp.var(axis=0, ddof=1)
            weighted_q = np.average(r2, weights=sy2)
            weighted_n, weighted_k = rule(weighted_q, h, dv, g)
            losses.append([
                exact_risk(xh, yh, beta, h, g, g),
                exact_risk(xh, yh, beta, h, n, k),
                exact_risk(xh, yh, beta, h, weighted_n, weighted_k),
            ])
        averaged = np.mean(losses, axis=0)
        for j, feature in enumerate(data["features"]):
            output.append(dict(feature=feature, delta_v=dv, splits=len(splits),
                               standardized_matched_direct_mse=averaged[0, j] / scale2[j],
                               standardized_raw_rule_mse=averaged[1, j] / scale2[j],
                               standardized_pilot_variance_weighted_rule_mse=averaged[2, j] / scale2[j],
                               raw_unit_matched_direct_mse=averaged[0, j],
                               raw_unit_raw_rule_mse=averaged[1, j],
                               raw_unit_pilot_variance_weighted_rule_mse=averaged[2, j]))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("Wrote", len(output), "feature-scenario rows", flush=True)


if __name__ == "__main__":
    main()

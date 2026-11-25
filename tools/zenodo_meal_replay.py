"""Exact finite-roster replay for the first lunch in Zenodo record 20699015."""

import argparse
import csv
from hashlib import md5
from io import BytesIO
from itertools import combinations
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXPECTED = {"female": "471ea8e0cbaeab7bcd9584773b39aa83",
            "male": "76bf0c5445cd1094cee8ecbf7abd9c7a"}
META = {"plate", "sample identification", "time point", "subject", "decimal sampling hours",
        "participant id"}
MALE_META = {"partcipant id_timepoint": "subject", "timepoint": "time point",
             "sampling hours": "decimal sampling hours"}


def parse_cell(cell, strings):
    value = cell.findtext("x:v", default="", namespaces=NS)
    if cell.get("t") == "s":
        return strings[int(value)]
    if cell.get("t") == "inlineStr":
        return "".join(x.text or "" for x in cell.findall(".//x:t", NS))
    return value


def measurement_rows(path, name):
    raw = path.read_bytes()
    if md5(raw).hexdigest() != EXPECTED[name]:
        raise ValueError(f"{name} does not match the repository checksum")
    with ZipFile(BytesIO(raw)) as workbook:
        shared = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = ["".join(t.text or "" for t in si.findall(".//x:t", NS)) for si in shared]
        candidates = []
        for sheet in sorted(p for p in workbook.namelist() if p.startswith("xl/worksheets/sheet") and p.endswith(".xml")):
            tree = ET.fromstring(workbook.read(sheet))
            rows = [{c.get("r").rstrip("0123456789"): parse_cell(c, strings)
                     for c in row.findall("x:c", NS)} for row in tree.findall(".//x:row", NS)]
            for idx, head in enumerate(rows[:10]):
                names = {MALE_META.get(name.strip().lower(), name.strip().lower()): column
                         for column, name in head.items() if name.strip()}
                if {"subject", "time point", "decimal sampling hours"} <= set(names):
                    features = {col: value.strip() for col, value in head.items()
                                if value.strip() and value.strip().lower() not in META | MALE_META.keys()}
                    candidates.append((sheet, names, features, rows[idx + 1:]))
                    break
    if len(candidates) != 1:
        print(name, "possible assay worksheets", [(s, len(f), len(r)) for s, _, f, r in candidates], flush=True)
        raise ValueError("Cannot uniquely identify the concentration worksheet")
    return candidates[0]


def to_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else math.nan
    except (TypeError, ValueError):
        return math.nan


def load(path, name):
    import numpy as np

    sheet, names, features, rows = measurement_rows(path, name)
    indexed = {}
    for row in rows:
        subject = row.get(names["subject"], "").strip()
        time = to_number(row.get(names["time point"]))
        hour = to_number(row.get(names["decimal sampling hours"]))
        if name == "male" and subject and math.isfinite(time):
            subject, sep, suffix = subject.rpartition("_")
            if not sep or to_number(suffix) != time:
                raise ValueError("Male combined identifier disagrees with timepoint")
            hour = round(hour * 24, 8)
        if subject and math.isfinite(hour):
            key = (time, hour)
            if key in indexed.setdefault(subject, {}):
                raise ValueError("Repeated subject/time key")
            indexed[subject][key] = [to_number(row.get(col)) for col in features]
    all_trajectories = sorted(indexed)
    schedule = sorted(set(key for times in indexed.values() for key in times))
    print(name, sheet, "participants", len(indexed), "time/hours", schedule, flush=True)
    if len(indexed) != 12:
        raise ValueError("Unexpected number of participants")
    matching = [key for key in schedule if key[1] in (12.0, 14.0, 16.0)]
    # Use the first occurrence when clock time is repeated on the next day.
    chosen = [min((key for key in matching if key[1] == hour), key=lambda key: key[0])
              for hour in (12.0, 14.0, 16.0)]
    if not all(chosen[i][0] < chosen[i + 1][0] for i in (0, 1)):
        raise ValueError("Cannot verify the first-day noon, 14:00 and 16:00 sequence")
    schedule_complete = [t for t in all_trajectories if all(key in indexed[t] for key in chosen)]
    candidate_values = np.asarray([[indexed[t][key] for key in chosen] for t in schedule_complete])
    available_counts = np.isfinite(candidate_values).sum(axis=2)
    print(name, "counts of valid concentrations in selected visits", sorted(map(tuple, available_counts)), flush=True)
    # Match the primary replays: an entirely unmeasured visit prevents a
    # participant from contributing to a complete reference trajectory.
    values = candidate_values[np.all(available_counts > 0, axis=1)]
    positive = np.all(np.isfinite(values), axis=(0, 1))
    log = np.log2(values[:, :, positive])
    x, y = log[:, 1] - log[:, 0], log[:, 2] - log[:, 0]
    varying = (x.std(axis=0) > 0) & (y.std(axis=0) > 0)
    print(name, "candidate/complete-positive/varying:", len(features), int(positive.sum()),
          int(varying.sum()), "selected records", chosen, flush=True)
    return x[:, varying], y[:, varying], len(features), int(positive.sum()), len(indexed)


def slope_and_r2(x, y):
    import numpy as np

    x0, y0 = x - x.mean(axis=0), y - y.mean(axis=0)
    xx, xy, yy = (x0 * x0).sum(axis=0), (x0 * y0).sum(axis=0), (y0 * y0).sum(axis=0)
    return (np.divide(xy, xx, out=np.zeros_like(xy), where=xx > 1e-14),
            np.divide(xy * xy, xx * yy, out=np.zeros_like(xy), where=xx * yy > 1e-20))


def replay(x, y, study, n_available, n_candidate, n_complete_features):
    import numpy as np

    m, g, h = 4, 6, len(y) - 4
    n_features = y.shape[1]
    if h <= g or n_features < 1:
        raise ValueError("Insufficient evaluable trajectories or features")
    scale2 = y.var(axis=0, ddof=1)
    source_q = slope_and_r2(x, y)[1]
    groups = []
    for dv in (2, 8):
        total, after_pilot = m * (3 + dv) + g * (2 + dv), g * (2 + dv)
        no_pilot_n = min(len(y), total // (2 + dv))
        without_pilot = 1 / no_pilot_n - 1 / len(y)
        candidates = [(n, k) for n in range(4, h + 1) for k in range(4, n + 1)
                      if 2 * n + (1 + dv) * k <= after_pilot]
        losses, decisions = [], []
        for pilot in combinations(range(len(y)), m):
            train = np.asarray(pilot)
            hold = np.asarray(sorted(set(range(len(y))) - set(pilot)))
            beta, pilot_r2 = slope_and_r2(x[train], y[train])
            risks = []
            for q in (pilot_r2.mean(), np.maximum(0, 1 - (1 - pilot_r2) * (m - 1) / (m - 2)).mean()):
                n, k = min(candidates, key=lambda nk: ((1 - q) / nk[1] + q / nk[0], -nk[1], -nk[0]))
                selected = (1 - q) / k + q / n < 1 / g - 1e-12
                vx = y[hold].var(axis=0, ddof=1)
                direct = np.mean(vx / scale2) * (1 / g - 1 / h)
                if selected:
                    ve = (y[hold] - beta * x[hold]).var(axis=0, ddof=1)
                    nested = np.mean((vx * (1 / n - 1 / h) + ve * (1 / k - 1 / n)) / scale2)
                else:
                    nested = direct
                risks.append((direct, nested, selected))
            losses.append([risks[0][0], risks[0][1], risks[1][1]])
            decisions.append([risks[0][2], risks[1][2]])
        loss = np.asarray(losses)
        selected = np.sum(decisions, axis=0)
        groups.append(dict(study=study, people_available=n_available, people_complete=len(y),
                           features_candidate=n_candidate, features_complete_positive=n_complete_features,
                           features_analyzed=n_features, median_feature_r2=float(np.median(source_q)),
                           pilot_m=m, direct_g=g, delta_v=dv, budget=total,
                           oracle_pool_size=h, exact_splits=len(loss),
                           raw_nested_count=int(selected[0]), adjusted_nested_count=int(selected[1]),
                           matched_direct_mse=float(loss[:, 0].mean()), raw_policy_mse=float(loss[:, 1].mean()),
                           adjusted_policy_mse=float(loss[:, 2].mean()),
                           raw_minus_matched_direct=float((loss[:, 1] - loss[:, 0]).mean()),
                           same_full_target_direct_mse=float(loss[:, 0].mean() * (h / len(y)) ** 2),
                           same_full_target_raw_mse=float(loss[:, 1].mean() * (h / len(y)) ** 2),
                           same_full_target_adjusted_mse=float(loss[:, 2].mean() * (h / len(y)) ** 2),
                           same_full_target_no_pilot_mse=without_pilot, no_pilot_n=no_pilot_n))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("zenodo_lunch_replay.csv"))
    args = parser.parse_args()
    results = []
    for cohort in EXPECTED:
        x, y, candidate, complete_features, n_participants = load(args.data / f"{cohort}.csv", cohort)
        results.extend(replay(x, y, cohort, n_participants, candidate, complete_features))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("Completed", len(results), "aggregate replay cells", flush=True)


if __name__ == "__main__":
    main()

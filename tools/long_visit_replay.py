"""Exploratory equal-budget replay in the existing ST001251 long-visit cohort."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

import frozen_benchmark as b


def load(path):
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header, factors = next(reader), next(reader)
        rows = [row for row in reader if row]
    if len(factors) != len(header) or any(len(row) != len(header) for row in rows):
        raise ValueError('Nonrectangular study table')
    visits = defaultdict(dict)
    for col, sample in enumerate(header[1:]):
        match = re.search(r'(\d+)_\s*(\d+)\s*_t_(minus_50|-?\d+)', sample)
        if match is None:
            raise ValueError('Unrecognized sample label')
        group, subject, visit = match.groups()
        if visit == 'minus_50':
            continue
        subject_key, visit = (group, subject), int(visit)
        if visit in visits[subject_key]:
            raise ValueError('Duplicate participant/visit')
        visits[subject_key][visit] = col
    values = np.array([[float(x) if x.strip() else np.nan for x in row[1:]] for row in rows]).T
    return visits, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    visits, values = load(args.source)
    print(f'Source SHA-256: {b.sha256(args.source)}; matrix {values.shape}; subjects {len(visits)}')
    output = []
    for early in (120, 240):
        times = (0, early, 480)
        subjects = [key for key, record in sorted(visits.items()) if all(t in record for t in times)]
        cube = values[[[visits[key][t] for t in times] for key in subjects]]
        available = (np.isfinite(cube) & (cube > 0)).all(axis=(0, 1))
        changes = np.log2(cube[:, 1:, available]) - np.log2(cube[:, 0:1, available])
        variable = np.all(np.std(changes, axis=0) > 1e-10 * np.maximum(1, np.max(abs(changes), axis=0)), axis=0)
        x, y = changes[:, 0, variable], changes[:, 1, variable]
        if len(subjects) < 18 or x.shape[1] == 0:
            raise ValueError('Insufficient participants or nonconstant markers for replay')
        print(f'Window 0/{early}/480 min: {len(subjects)} participants; {x.shape[1]} retained markers', flush=True)
        split_rng = np.random.default_rng(1861251)
        splits = [split_rng.permutation(len(subjects)) for _ in range(b.N_OUTER)]
        for dv in (0.98, 2.0, 8.0):
            for g in b.G_VALUES:
                data = {'X': x, 'Y': y}
                results = [b.simulate_split(data, perm[:b.PILOT_M], perm[b.PILOT_M:], dv, g,
                           np.random.default_rng(2861251 + i * 1000 + int(dv * 100) + g))
                           for i, perm in enumerate(splits)]
                output.append({
                    'auxiliary_min': early, 'late_min': 480, 'participants': len(subjects),
                    'eligible_features': x.shape[1], 'delta_v': dv, 'g': g, 'splits': len(results),
                    'mean_pilot_r2': fmean(r['mean_pilot_r2'] for r in results),
                    'policy_borrow_fraction': fmean(r['decision'] == 'borrow' for r in results),
                    'no_pilot_census_fraction': fmean(r['n_no_pilot'] == r['n_eval'] for r in results),
                    **{f'{arm}_mse': fmean(r['scores'][arm]['mse'] for r in results)
                       for arm in results[0]['scores']},
                })
                print(output[-1], flush=True)
    b.write_csv(args.output, output)


if __name__ == '__main__':
    main()

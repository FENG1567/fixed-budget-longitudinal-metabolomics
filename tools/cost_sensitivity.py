#!/usr/bin/env python3
"""Deterministic cost sensitivity for the IJE two-phase design criterion.

All costs are assay-equivalent units. This script uses no study data and no
simulation; it evaluates analytic and finite-integer design boundaries.
"""
import argparse
import csv
import math
from pathlib import Path


def continuous_ratio(alpha, q, A, C, D):
    return ((A + C * alpha) / D) * (((1.0 - q) / alpha) + q)


def continuous_min(q, A, C, D):
    # The formula is minimized on alpha in (0, 1]. For A=0 the infimum may
    # occur only as alpha approaches zero; report that limit separately.
    if A == 0.0:
        if q == 0.0:
            alpha = 1.0
            return C / D, alpha
        return C * (1.0 - q) / D, 0.0
    if q <= 0.0:
        alpha = 1.0
    elif q >= 1.0:
        alpha = 0.0
        return A / D, alpha
    else:
        alpha = min(1.0, math.sqrt(A * (1.0 - q) / (C * q)))
    return continuous_ratio(alpha, q, A, C, D), alpha


def continuous_threshold(A, C, D):
    if A == 0.0:
        # The infimum is reached only as alpha -> 0 when rho2 > 0.
        qstar = max(0.0, 1.0 - D / C)
        return qstar, 0.0, qstar < 1.0
    if A >= D:
        return 1.0, 0.0, False
    if A + C < D - 1e-12:
        return 0.0, 1.0, True
    if abs(A + C - D) <= 1e-12:
        # At equality alpha=1 ties at rho2=0; the optimum begins to improve
        # only after q exceeds A/(A+C).
        return A / (A + C), 1.0, True
    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = (lo + hi) / 2.0
        value, _ = continuous_min(mid, A, C, D)
        if value < 1.0:
            hi = mid
        else:
            lo = mid
    qstar = (lo + hi) / 2.0
    _, alpha = continuous_min(qstar, A, C, D)
    return qstar, alpha, True


def integer_threshold(A, C, D, B, H=13, min_k=4):
    direct_n = min(H, math.floor(B / D + 1e-10))
    if direct_n < 1 or direct_n >= H:
        return None, False, None, None, None, B, direct_n
    all_allocations = []
    improving_candidates = []
    for N in range(1, H + 1):
        for k in range(min_k, N + 1):
            cost = A * N + C * k
            if cost > B + 1e-10:
                continue
            if k == N:
                qcrit = 0.0 if N > direct_n else (None if N == direct_n else 2.0)
            else:
                denom = (1.0 / k) - (1.0 / N)
                qcrit = ((1.0 / k) - (1.0 / direct_n)) / denom
                qcrit = max(0.0, qcrit)
            allocation = (qcrit, N, k, cost)
            all_allocations.append(allocation)
            if qcrit is not None and qcrit <= 1.0 + 1e-12:
                improving_candidates.append(allocation)
    if not all_allocations:
        return None, False, None, None, None, B, direct_n
    if not improving_candidates:
        qstar, possible, Nstar, kstar = None, False, None, None
    else:
        best = min(improving_candidates, key=lambda x: (x[0], x[3], x[1], x[2]))
        qstar, Nstar, kstar, _ = best
        possible = qstar < 1.0 - 1e-10
    # A strict gain at some q in [0,1] requires a crossing below one.
    q_eval = 0.82
    def finite_var(x):
        return ((1.0 - q_eval) / x[2] + q_eval / x[1] - 1.0 / H)
    best_eval = min(all_allocations, key=lambda x: (finite_var(x), x[3], x[1], x[2]))
    direct_var = 1.0 / direct_n - 1.0 / H
    ratio_eval = finite_var(best_eval) / direct_var
    return qstar, possible, Nstar, kstar, (best_eval[1], best_eval[2], best_eval[3], ratio_eval), B, direct_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for aux_status in ('new_auxiliary', 'auxiliary_already_available'):
        for h in (0, 1, 2, 4):
            for visit_mode, visit_fee in (('in_visit', 0), ('return_visit', 2)):
                for d in (0, 2, 8):
                    if aux_status == 'new_auxiliary':
                        A = 2 + h       # baseline + early assays + shared recruitment
                        C = 1 + d + visit_fee  # late assay + incremental visit cost
                        D = 2 + d + visit_fee + h  # baseline + late + recruitment
                        effective_h = h
                    else:
                        # Same finite roster (H=13) already has baseline + early
                        # results. Their recruitment and assays are sunk; h is a
                        # common sunk enrollment cost outside this marginal budget.
                        A = 0
                        C = 1 + d + visit_fee
                        D = 1 + d + visit_fee
                        effective_h = 0
                    qstar, alpha_star, cont_possible = continuous_threshold(A, C, D)
                    ratio82, alpha82 = continuous_min(0.82, A, C, D)
                    B_matched = 12 * D
                    qi, int_possible, Ni, ki, eval82, B, g_matched = integer_threshold(A, C, D, B_matched)
                    if eval82:
                        N82, k82, cost82, int_ratio82 = eval82
                    else:
                        N82 = k82 = cost82 = int_ratio82 = ''
                    qfixed, fixed_possible, Nfixed, kfixed, eval_fixed, Bfixed, gfixed = integer_threshold(A, C, D, 48.0)
                    if eval_fixed:
                        Nfixed82, kfixed82, costfixed82, ratiofixed82 = eval_fixed
                    else:
                        Nfixed82 = kfixed82 = costfixed82 = ratiofixed82 = ''
                    rows.append({
                        'auxiliary_status': aux_status,
                        'h_scenario_assay_units': h,
                        'effective_incremental_h': effective_h,
                        'visit_mode': visit_mode,
                        'return_visit_fee_assumption': visit_fee,
                        'incremental_visit_cost_d': d,
                        'A_early_per_N': A,
                        'C_late_per_k': C,
                        'D_direct_per_person': D,
                        'continuous_rho2_threshold': qstar,
                        'continuous_strict_gain_possible': cont_possible,
                        'continuous_alpha_at_boundary': alpha_star,
                        'continuous_min_ratio_at_rho2_0_82': ratio82,
                        'continuous_alpha_opt_at_rho2_0_82': alpha82,
                        'matched_budget_W_equals_12D': B,
                        'integer_pool_cap_H': 13,
                        'matched_direct_g': g_matched,
                        'matched_min_k': 4,
                        'matched_integer_rho2_threshold': '' if qi is None else qi,
                        'matched_integer_strict_gain_possible': int_possible,
                        'matched_integer_boundary_N': '' if Ni is None else Ni,
                        'matched_integer_boundary_k': '' if ki is None else ki,
                        'matched_opt_N_at_rho2_0_82': N82,
                        'matched_opt_k_at_rho2_0_82': k82,
                        'matched_opt_cost_at_rho2_0_82': cost82,
                        'matched_finite_roster_MSE_ratio_at_rho2_0_82': int_ratio82,
                        'fixed_budget_W': Bfixed,
                        'fixed_direct_g': gfixed,
                        'fixed_min_k': 4,
                        'fixed_integer_rho2_threshold': '' if qfixed is None else qfixed,
                        'fixed_integer_strict_gain_possible': fixed_possible,
                        'fixed_integer_boundary_N': '' if Nfixed is None else Nfixed,
                        'fixed_integer_boundary_k': '' if kfixed is None else kfixed,
                        'fixed_opt_N_at_rho2_0_82': Nfixed82,
                        'fixed_opt_k_at_rho2_0_82': kfixed82,
                        'fixed_opt_cost_at_rho2_0_82': costfixed82,
                        'fixed_finite_roster_MSE_ratio_at_rho2_0_82': ratiofixed82,
                    })
    csv_path = out / 'cost_scenarios.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    thresholds = [r['continuous_rho2_threshold'] for r in rows]
    int_thresholds = [float(r['matched_integer_rho2_threshold']) for r in rows if r['matched_integer_rho2_threshold'] != '']
    fixed_thresholds = [float(r['fixed_integer_rho2_threshold']) for r in rows if r['fixed_integer_rho2_threshold'] != '']
    print(f'Wrote {csv_path} ({len(rows)} scenarios)')
    print(f'Continuous rho2 threshold range: {min(thresholds):.6f} to {max(thresholds):.6f}')
    print(f'Matched-budget integer rho2 threshold range: {min(int_thresholds):.6f} to {max(int_thresholds):.6f}')
    print(f'Fixed-W=48 integer rho2 threshold range: {min(fixed_thresholds):.6f} to {max(fixed_thresholds):.6f}')
    print(f'Continuous strict gain possible: {sum(bool(r["continuous_strict_gain_possible"]) for r in rows)}/{len(rows)}')
    print(f'Matched-budget integer strict gain possible: {sum(bool(r["matched_integer_strict_gain_possible"]) for r in rows)}/{len(rows)}')
    print(f'Fixed-W strict gain possible: {sum(bool(r["fixed_integer_strict_gain_possible"]) for r in rows)}/{len(rows)}')
    print('Theoretical check at original costs A=2,C=3,D=4:')
    qstar, alpha, possible = continuous_threshold(2, 3, 4)
    print(f'  rho2*={qstar:.6f}; alpha*={alpha:.6f}; strict_gain={possible}')


if __name__ == '__main__':
    main()

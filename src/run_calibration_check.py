"""
Calibration control  ->  Section 3.10 and Table 10.

The objection this answers
--------------------------
Section 3.4 argues that the raw convex coefficients cannot be read as
importance scores because the three experts differ in the sharpness of their
posteriors.  A natural reply is that unequal sharpness is simply
miscalibration, and that temperature scaling should be applied before the
coefficients are interpreted.

This script tests that reply directly.  For each re-split it fits one
temperature per expert on the validation split by minimising the negative
log-likelihood, re-runs the Dirichlet search on the calibrated posteriors,
and compares both attributions -- raw coefficient and effective
contribution -- against the same single-group ablation ground truth, under
both conditions.

Temperature is applied to the Platt-scaled posteriors as p ** (1 / T),
renormalised, which is temperature scaling on the log-probabilities.  T < 1
sharpens the distribution, T > 1 flattens it.

The re-splits are constructed exactly as in run_stability.py, so the two
scripts are directly comparable.

This script fits six SVMs per split, all with Platt scaling.  Expect roughly
20-40 minutes on a single commodity CPU core for the full ten splits; use
--splits to run fewer while testing.

Usage:
    python src/run_calibration_check.py
    python src/run_calibration_check.py --splits 3
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from common import (
    GROUPS,
    RESULT_DIR,
    SEED,
    banner,
    feature_matrix,
    load_all,
    make_svm,
)

N_TRIALS = 1000
N_BINS = 15
EPS = 1e-12


def fuse(alpha, posteriors):
    stacked = sum(a * P for a, P in zip(alpha, posteriors))
    return stacked / (stacked.sum(axis=1, keepdims=True) + EPS)


def temper(P, T):
    """Temperature scaling on the log-probabilities: p ** (1 / T), renormalised."""
    scaled = np.exp(np.log(np.clip(P, EPS, 1.0)) / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(P, y):
    """One temperature per expert, fitted by minimising validation NLL."""
    index = np.arange(len(y))

    def nll(T):
        return -np.log(np.clip(temper(P, T)[index, y], EPS, 1.0)).mean()

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def ece(P, y, n_bins=N_BINS):
    """Expected calibration error of the predicted-class posterior."""
    pred = P.argmax(axis=1)
    conf = P.max(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return total


def entropy(P):
    return float((-(P * np.log(np.clip(P, EPS, 1.0))).sum(axis=1)).mean())


def search_alpha(posteriors_val, y_val):
    rng = np.random.default_rng(SEED)
    best_alpha, best_score = None, -np.inf
    for _ in range(N_TRIALS):
        candidate = rng.dirichlet([1.0, 1.0, 1.0])
        score = f1_score(
            y_val, fuse(candidate, posteriors_val).argmax(axis=1), average="macro"
        )
        if score > best_score:
            best_alpha, best_score = candidate, score
    return best_alpha, best_score


def attribution(alpha, posteriors_test, y_test):
    index = np.arange(len(y_test))
    effective = np.array(
        [a * P[index, y_test].mean() for a, P in zip(alpha, posteriors_test)]
    )
    return effective


def ordering(values):
    return tuple(GROUPS[i] for i in np.argsort(values)[::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=10)
    args = parser.parse_args()

    train, val, test, columns = load_all()
    pooled = pd.concat([train, val, test], ignore_index=True)
    y_all = pooled["label_id"].values
    group_matrices = {g: feature_matrix(pooled, columns, (g,)) for g in GROUPS}

    banner(f"Section 3.10 - calibration control over {args.splits} re-splits")

    per_expert, per_split = [], []
    for seed in range(args.splits):
        index = np.arange(len(y_all))
        idx_train, idx_rest = train_test_split(
            index, test_size=0.30, stratify=y_all, random_state=seed
        )
        idx_val, idx_test = train_test_split(
            idx_rest, test_size=0.50, stratify=y_all[idx_rest], random_state=seed
        )
        y_val, y_test = y_all[idx_val], y_all[idx_test]

        raw_val, raw_test, single_f1, temperatures = [], [], [], []
        for group in GROUPS:
            expert = make_svm(probability=True)
            expert.fit(group_matrices[group][idx_train], y_all[idx_train])
            P_val = expert.predict_proba(group_matrices[group][idx_val])
            P_test = expert.predict_proba(group_matrices[group][idx_test])
            raw_val.append(P_val)
            raw_test.append(P_test)
            single_f1.append(
                f1_score(y_test, P_test.argmax(axis=1), average="macro") * 100
            )
            temperatures.append(fit_temperature(P_val, y_val))

        cal_val = [temper(P, T) for P, T in zip(raw_val, temperatures)]
        cal_test = [temper(P, T) for P, T in zip(raw_test, temperatures)]

        for j, group in enumerate(GROUPS):
            per_expert.append(
                dict(
                    split=seed,
                    expert=group,
                    temperature=temperatures[j],
                    single_group_macro_f1=single_f1[j],
                    ece_before=ece(raw_test[j], y_test),
                    ece_after=ece(cal_test[j], y_test),
                    mean_max_p_before=raw_test[j].max(axis=1).mean(),
                    mean_max_p_after=cal_test[j].max(axis=1).mean(),
                    entropy_before=entropy(raw_test[j]),
                    entropy_after=entropy(cal_test[j]),
                )
            )

        truth = ordering(single_f1)
        record = dict(split=seed, ablation_order=" > ".join(truth))
        for condition, P_val_set, P_test_set in (
            ("uncalibrated", raw_val, raw_test),
            ("calibrated", cal_val, cal_test),
        ):
            alpha, _ = search_alpha(P_val_set, y_val)
            effective = attribution(alpha, P_test_set, y_test)
            fused = fuse(alpha, P_test_set).argmax(axis=1)
            record[f"{condition}_fusion_macro_f1"] = (
                f1_score(y_test, fused, average="macro") * 100
            )
            record[f"{condition}_raw_recovers"] = ordering(alpha) == truth
            record[f"{condition}_effective_recovers"] = ordering(effective) == truth
            record[f"{condition}_raw_spearman"] = spearmanr(alpha, single_f1).statistic
            record[f"{condition}_effective_spearman"] = spearmanr(
                effective, single_f1
            ).statistic
            for name, a, e in zip(GROUPS, alpha, effective / effective.sum()):
                record[f"{condition}_alpha_{name}"] = a
                record[f"{condition}_share_{name}"] = e * 100

        per_split.append(record)
        print(
            f"split {seed}: T = "
            + ", ".join(f"{g} {t:.3f}" for g, t in zip(GROUPS, temperatures))
            + f" | raw recovers {record['uncalibrated_raw_recovers']}"
            f" -> {record['calibrated_raw_recovers']}"
            f" | effective {record['uncalibrated_effective_recovers']}"
            f" -> {record['calibrated_effective_recovers']}"
        )

    expert_frame = pd.DataFrame(per_expert)
    split_frame = pd.DataFrame(per_split)
    expert_frame.to_csv(RESULT_DIR / "table10_calibration_per_expert.csv", index=False)
    split_frame.to_csv(RESULT_DIR / "table10_calibration_attribution.csv", index=False)

    # ---- Table 10 ---------------------------------------------------------
    banner("Table 10 - per-expert effect of temperature scaling")
    rows = []
    for group in GROUPS:
        block = expert_frame[expert_frame["expert"] == group]
        rows.append(
            dict(
                expert=group,
                T=f"{block['temperature'].mean():.3f} +/- {block['temperature'].std():.3f}",
                ECE=f"{block['ece_before'].mean():.3f} -> {block['ece_after'].mean():.3f}",
                mean_max_P=f"{block['mean_max_p_before'].mean():.3f} -> "
                f"{block['mean_max_p_after'].mean():.3f}",
                entropy=f"{block['entropy_before'].mean():.3f} -> "
                f"{block['entropy_after'].mean():.3f}",
            )
        )
    table10 = pd.DataFrame(rows)
    print(table10.to_string(index=False))
    table10.to_csv(RESULT_DIR / "table10_calibration.csv", index=False)

    gap_before = (
        expert_frame[expert_frame.expert == "color"]["mean_max_p_before"].mean()
        - expert_frame[expert_frame.expert == "shape"]["mean_max_p_before"].mean()
    )
    gap_after = (
        expert_frame[expert_frame.expert == "color"]["mean_max_p_after"].mean()
        - expert_frame[expert_frame.expert == "shape"]["mean_max_p_after"].mean()
    )
    print(
        f"\nsharpness gap (color - shape): {gap_before:.3f} -> {gap_after:.3f}"
        f"   (closed by {gap_before - gap_after:.3f})"
    )

    # ---- the argument -----------------------------------------------------
    banner("Section 3.10 - does calibration remove the need for the correction?")
    n = len(split_frame)
    for condition in ("uncalibrated", "calibrated"):
        print(
            f"{condition:13s}: raw recovers the ablation ordering in "
            f"{int(split_frame[f'{condition}_raw_recovers'].sum())}/{n} splits "
            f"(Spearman {split_frame[f'{condition}_raw_spearman'].mean():.2f}), "
            f"effective in "
            f"{int(split_frame[f'{condition}_effective_recovers'].sum())}/{n} "
            f"(Spearman {split_frame[f'{condition}_effective_spearman'].mean():.2f}), "
            f"fused test Macro-F1 "
            f"{split_frame[f'{condition}_fusion_macro_f1'].mean():.2f}%"
        )
    print(f"\nwrote {RESULT_DIR / 'table10_calibration.csv'}")


if __name__ == "__main__":
    main()

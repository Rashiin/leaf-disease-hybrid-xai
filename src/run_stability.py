"""
Stability across ten independent stratified re-splits  ->  Table 8
(Section 3.8), plus the repeated-split selective-prediction figures quoted
at the end of Section 3.9.

The three splits shipped in data/ are pooled back into the full 20,638-image
feature set and re-partitioned 70/15/15 with seeds 0-9.  No hyperparameter,
threshold or design choice changes between runs.

This is the slowest script in the repository: it fits four SVMs per split,
three of them with Platt scaling.  Expect roughly 30-60 minutes on a single
commodity CPU core.  Use --splits to run fewer partitions while testing.

Usage:
    python src/run_stability.py
    python src/run_stability.py --splits 3
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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


def fuse(alpha, posteriors):
    stacked = sum(a * P for a, P in zip(alpha, posteriors))
    return stacked / (stacked.sum(axis=1, keepdims=True) + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=10, help="number of re-splits")
    args = parser.parse_args()

    train, val, test, columns = load_all()
    pooled = pd.concat([train, val, test], ignore_index=True)
    y_all = pooled["label_id"].values
    group_matrices = {g: feature_matrix(pooled, columns, (g,)) for g in GROUPS}
    X_all = feature_matrix(pooled, columns)

    banner(f"Table 8 - {args.splits} independent stratified 70/15/15 re-splits")
    print(f"pooled dataset: {X_all.shape[0]} images, {X_all.shape[1]} features\n")

    records = []
    for seed in range(args.splits):
        index = np.arange(len(y_all))
        idx_train, idx_rest = train_test_split(
            index, test_size=0.30, stratify=y_all, random_state=seed
        )
        idx_val, idx_test = train_test_split(
            idx_rest, test_size=0.50, stratify=y_all[idx_rest], random_state=seed
        )

        # baseline (calibrated so the same run also yields the reject option)
        baseline = make_svm(probability=True)
        baseline.fit(X_all[idx_train], y_all[idx_train])
        pred_test = baseline.predict(X_all[idx_test])
        y_test = y_all[idx_test]
        correct = pred_test == y_test

        record = {
            "split": seed,
            "baseline_accuracy": accuracy_score(y_test, pred_test) * 100,
            "baseline_macro_f1": f1_score(y_test, pred_test, average="macro") * 100,
            "support_vectors": baseline.named_steps["clf"].support_vectors_.shape[0],
        }

        # reject option, thresholds calibrated on this split's validation set
        posterior_val = baseline.predict_proba(X_all[idx_val])
        pred_val = baseline.predict(X_all[idx_val])
        conf_val = posterior_val[np.arange(len(pred_val)), pred_val]
        posterior_test = baseline.predict_proba(X_all[idx_test])
        conf_test = posterior_test[np.arange(len(pred_test)), pred_test]

        record["error_auroc"] = roc_auc_score((~correct).astype(int), -conf_test)
        for target in (0.95, 0.90):
            threshold = np.quantile(conf_val, 1 - target)
            keep = conf_test >= threshold
            present = np.unique(y_test[keep])
            tag = int(target * 100)
            record[f"coverage_{tag}"] = keep.mean() * 100
            record[f"accuracy_{tag}"] = accuracy_score(y_test[keep], pred_test[keep]) * 100
            record[f"macro_f1_{tag}"] = (
                f1_score(y_test[keep], pred_test[keep], average="macro", labels=present) * 100
            )
            record[f"errors_deferred_{tag}"] = (
                (~correct & ~keep).sum() / max((~correct).sum(), 1) * 100
            )

        # three experts, Dirichlet search, fusion
        posteriors_val, posteriors_test = [], []
        for group in GROUPS:
            expert = make_svm(probability=True)
            expert.fit(group_matrices[group][idx_train], y_all[idx_train])
            posteriors_val.append(expert.predict_proba(group_matrices[group][idx_val]))
            posteriors_test.append(expert.predict_proba(group_matrices[group][idx_test]))

        rng = np.random.default_rng(SEED)
        best_alpha, best_score = None, -np.inf
        y_val = y_all[idx_val]
        for _ in range(N_TRIALS):
            candidate = rng.dirichlet([1.0, 1.0, 1.0])
            score = f1_score(
                y_val, fuse(candidate, posteriors_val).argmax(axis=1), average="macro"
            )
            if score > best_score:
                best_alpha, best_score = candidate, score

        fused = fuse(best_alpha, posteriors_test).argmax(axis=1)
        uniform = fuse(np.repeat(1 / 3, 3), posteriors_test).argmax(axis=1)
        record["fusion_macro_f1"] = f1_score(y_test, fused, average="macro") * 100
        record["uniform_macro_f1"] = f1_score(y_test, uniform, average="macro") * 100

        index_test = np.arange(len(y_test))
        effective = np.array(
            [a * P[index_test, y_test].mean() for a, P in zip(best_alpha, posteriors_test)]
        )
        shares = effective / effective.sum()
        for name, weight, share in zip(GROUPS, best_alpha, shares):
            record[f"alpha_{name}"] = weight
            record[f"share_{name}"] = share * 100

        records.append(record)
        print(
            f"split {seed}: baseline {record['baseline_accuracy']:.2f}/"
            f"{record['baseline_macro_f1']:.2f}  fusion {record['fusion_macro_f1']:.2f}  "
            f"uniform {record['uniform_macro_f1']:.2f}  "
            f"AUROC {record['error_auroc']:.3f}  acc@90 {record['accuracy_90']:.2f}"
        )

    frame = pd.DataFrame(records)
    frame.to_csv(RESULT_DIR / "table8_stability.csv", index=False)

    banner("Mean +/- s.d. over the re-splits")
    for column in (
        "baseline_accuracy",
        "baseline_macro_f1",
        "fusion_macro_f1",
        "uniform_macro_f1",
        "support_vectors",
        "error_auroc",
        "coverage_90",
        "accuracy_90",
        "macro_f1_90",
        "errors_deferred_90",
    ):
        print(f"{column:20s}: {frame[column].mean():.3f} +/- {frame[column].std():.3f}")

    wins = int((frame["fusion_macro_f1"] > frame["uniform_macro_f1"]).sum())
    margin = frame["fusion_macro_f1"] - frame["uniform_macro_f1"]
    print(
        f"\nfusion beats the uniform control in {wins}/{len(frame)} splits, "
        f"by {margin.mean():.2f} +/- {margin.std():.2f} points"
    )
    print(f"\nwrote {RESULT_DIR / 'table8_stability.csv'}")


if __name__ == "__main__":
    main()

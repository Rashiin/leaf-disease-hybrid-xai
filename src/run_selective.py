"""
Selective prediction with a reject option  ->  Table 9 and Figure 4
(Section 3.9).

The confidence score is the Platt-calibrated posterior that the baseline
SVM assigns to its own predicted class.  Enabling `probability=True` does
not change `predict`, so the full-coverage row of Table 9 is identical to
the baseline row of Table 4 by construction.

Thresholds are chosen as quantiles of the *validation* confidence at the
target coverage and then applied unchanged to the test set, so no test
information enters the operating point.

Usage:  python src/run_selective.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from common import (
    FIGURE_DIR,
    RESULT_DIR,
    banner,
    class_names,
    feature_matrix,
    load_all,
    make_svm,
)

TARGETS = (1.00, 0.95, 0.90, 0.85, 0.80)

# The five categories that Section 3.3 identifies as the dominant confusion
# structure: the potato/tomato blights plus the Target_Spot / Spider_mites pair.
CONFUSION_FAMILY = (
    "Potato___Late_blight",
    "Tomato_Late_blight",
    "Tomato_Early_blight",
    "Tomato__Target_Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
)


def confidence(model, X):
    """Posterior assigned to the predicted class."""
    posterior = model.predict_proba(X)
    prediction = model.predict(X)
    return posterior[np.arange(len(prediction)), prediction], prediction


def main():
    train, val, test, columns = load_all()
    names = class_names(train)

    X_train = feature_matrix(train, columns)
    X_val = feature_matrix(val, columns)
    X_test = feature_matrix(test, columns)
    y_train = train["label_id"].values
    y_val = val["label_id"].values
    y_test = test["label_id"].values

    banner("Training the baseline with calibrated posteriors")
    model = make_svm(probability=True)
    model.fit(X_train, y_train)

    conf_val, _ = confidence(model, X_val)
    conf_test, pred_test = confidence(model, X_test)
    correct = pred_test == y_test
    n_errors = int((~correct).sum())

    print(f"test accuracy  : {accuracy_score(y_test, pred_test) * 100:.2f}%")
    print(f"test Macro-F1  : {f1_score(y_test, pred_test, average='macro') * 100:.2f}%")
    print(f"test errors    : {n_errors}")

    auroc = roc_auc_score((~correct).astype(int), -conf_test)
    print(f"error-detection AUROC: {auroc:.3f}")

    # ---- Table 9 ----------------------------------------------------------
    banner("Table 9 - selective prediction")
    rows = []
    for target in TARGETS:
        threshold = -np.inf if target >= 0.999 else np.quantile(conf_val, 1 - target)
        keep = conf_test >= threshold
        present = np.unique(y_test[keep])
        deferred_errors = int((~correct & ~keep).sum())
        rows.append(
            {
                "Target cov. (%)": round(target * 100),
                "Test cov. (%)": round(keep.mean() * 100, 2),
                "Accuracy (%)": round(accuracy_score(y_test[keep], pred_test[keep]) * 100, 2),
                "Macro-F1 (%)": round(
                    f1_score(y_test[keep], pred_test[keep], average="macro", labels=present) * 100,
                    2,
                ),
                "Errors retained": int((~correct[keep]).sum()),
                "Errors deferred (%)": round(deferred_errors / n_errors * 100, 1),
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(RESULT_DIR / "table9_selective.csv", index=False)

    # ---- where the abstentions land --------------------------------------
    banner("Section 3.9 - composition of the deferred set at 90% target coverage")
    threshold = np.quantile(conf_val, 0.10)
    keep = conf_test >= threshold

    frame = pd.DataFrame(
        {
            "label": [names[i] for i in y_test],
            "deferred": ~keep,
            "error": ~correct,
        }
    )
    grouped = frame.groupby("label").agg(
        n=("label", "size"), deferred=("deferred", "sum"), errors=("error", "sum")
    )
    deferral_rate = grouped["deferred"] / grouped["n"] * 100
    error_rate = grouped["errors"] / grouped["n"] * 100
    rho, pvalue = spearmanr(deferral_rate, error_rate)

    grouped["deferral_rate_%"] = deferral_rate.round(1)
    grouped["error_rate_%"] = error_rate.round(1)
    grouped = grouped.sort_values("deferral_rate_%", ascending=False)
    print(grouped.to_string())
    grouped.to_csv(RESULT_DIR / "section39_deferral_by_class.csv")

    print(f"\nSpearman(deferral rate, error rate) = {rho:.3f}  (p = {pvalue:.4f})")

    family = grouped.loc[[c for c in CONFUSION_FAMILY if c in grouped.index]]
    print(
        f"Section-3.3 confusion family: {family['n'].sum() / grouped['n'].sum() * 100:.1f}% "
        f"of the test set but {family['deferred'].sum() / grouped['deferred'].sum() * 100:.1f}% "
        f"of all deferrals"
    )

    # Potato___Late_blight, the weakest class of Table 6
    target_id = names.index("Potato___Late_blight")
    for description, mask in (("full coverage", np.ones(len(y_test), bool)), ("90% coverage", keep)):
        p, t = pred_test[mask], y_test[mask]
        tp = int(((p == target_id) & (t == target_id)).sum())
        fp = int(((p == target_id) & (t != target_id)).sum())
        fn = int(((p != target_id) & (t == target_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"Potato___Late_blight @ {description:13s}: "
            f"precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}  (FP={fp})"
        )

    # ---- Figure 4 ---------------------------------------------------------
    sweep = np.arange(1.00, 0.68, -0.01)
    coverage, accuracy, macro_f1, remaining = [], [], [], []
    for target in sweep:
        threshold = -np.inf if target >= 0.999 else np.quantile(conf_val, 1 - target)
        mask = conf_test >= threshold
        if mask.sum() < 50:
            continue
        present = np.unique(y_test[mask])
        coverage.append(mask.mean() * 100)
        accuracy.append(accuracy_score(y_test[mask], pred_test[mask]) * 100)
        macro_f1.append(
            f1_score(y_test[mask], pred_test[mask], average="macro", labels=present) * 100
        )
        remaining.append((~correct[mask]).sum() / n_errors * 100)

    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.14), dpi=300)

    axes[0].plot(coverage, accuracy, "-", color="#1f4e79", lw=1.6, label="Accuracy")
    axes[0].plot(coverage, macro_f1, "--", color="#c55a11", lw=1.6, label="Macro-F1")
    axes[0].set_ylabel("Performance on\naccepted samples (%)", fontsize=7)
    axes[0].set_title("(a) Reliability-coverage trade-off", fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    axes[0].set_ylim(97.0, 100.0)

    axes[1].plot(coverage, remaining, "-", color="#548235", lw=1.6)
    axes[1].fill_between(coverage, remaining, alpha=0.15, color="#548235")
    axes[1].set_ylabel(f"Remaining errors\n(% of {n_errors})", fontsize=7)
    axes[1].set_title("(b) Error mass retained", fontsize=7.5)
    axes[1].set_ylim(0, 105)

    for ax in axes:
        ax.set_xlabel("Coverage (%)", fontsize=7)
        ax.invert_xaxis()
        ax.grid(alpha=0.3, lw=0.4)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIGURE_DIR / "figure4_selective.png", bbox_inches="tight")
    print(f"\nwrote {FIGURE_DIR / 'figure4_selective.png'}")


if __name__ == "__main__":
    main()

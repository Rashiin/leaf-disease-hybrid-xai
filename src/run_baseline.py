"""
Baseline hybrid RBF-SVM  ->  Table 4 (baseline row), Table 6, Figure 1.

Usage:  python src/run_baseline.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from common import (
    FIGURE_DIR,
    RESULT_DIR,
    banner,
    class_names,
    feature_matrix,
    load_all,
    make_svm,
)


def main():
    train, val, test, columns = load_all()
    names = class_names(train)

    X_train = feature_matrix(train, columns)
    X_val = feature_matrix(val, columns)
    X_test = feature_matrix(test, columns)
    y_train = train["label_id"].values
    y_val = val["label_id"].values
    y_test = test["label_id"].values

    banner("Baseline SVM (all 87 features, no PCA)")
    print(f"feature dimensionality : {X_train.shape[1]}")
    print(f"train / val / test     : {len(y_train)} / {len(y_val)} / {len(y_test)}")

    model = make_svm()
    model.fit(X_train, y_train)

    rows = []
    for name, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred) * 100
        macro_f1 = f1_score(y, pred, average="macro") * 100
        rows.append(dict(split=name, accuracy=accuracy, macro_f1=macro_f1))
        print(f"{name:11s}: accuracy {accuracy:.2f}%   Macro-F1 {macro_f1:.2f}%")

    n_sv = model.named_steps["clf"].support_vectors_.shape[0]
    print(f"\nsupport vectors: {n_sv}  "
          f"(~{n_sv * X_train.shape[1] / 1e5:.1f}e5 stored components)")

    pd.DataFrame(rows).to_csv(RESULT_DIR / "table4_baseline.csv", index=False)

    # ---- Table 6: per-class precision / recall / F1 / support --------------
    banner("Table 6 - per-class performance on the held-out test set")
    pred_test = model.predict(X_test)
    report = classification_report(
        y_test, pred_test, target_names=names, digits=3, output_dict=True, zero_division=0
    )
    per_class = (
        pd.DataFrame(report)
        .transpose()
        .loc[names, ["precision", "recall", "f1-score", "support"]]
        .round({"precision": 3, "recall": 3, "f1-score": 3})
    )
    per_class["support"] = per_class["support"].astype(int)
    print(per_class.to_string())
    per_class.to_csv(RESULT_DIR / "table6_per_class.csv")

    # ---- Figure 1: confusion matrix ---------------------------------------
    cm = confusion_matrix(y_test, pred_test)
    short = [
        n.replace("Tomato_", "Tomato ")
        .replace("Potato___", "Potato ")
        .replace("Pepper__bell___", "Pepper ")
        .replace("_", " ")
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=200)
    masked = np.ma.masked_where(cm == 0, cm)
    palette = plt.cm.Blues.copy()
    palette.set_bad("white")
    image = ax.imshow(masked, cmap=palette, norm=LogNorm(vmin=1, vmax=cm.max()))

    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(
                    j, i, cm[i, j],
                    ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() * 0.25 else "black",
                )

    fig.colorbar(image, ax=ax, label="Test images (log scale)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_confusion_matrix.png", bbox_inches="tight")
    print(f"\nwrote {FIGURE_DIR / 'figure1_confusion_matrix.png'}")


if __name__ == "__main__":
    main()

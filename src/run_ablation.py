"""
Feature-group ablation, with and without PCA  ->  Table 5.

Each configuration is trained from scratch on the same partition with the
same fixed hyperparameters, so the comparison reflects the information
content of the feature subsets rather than a per-subset hyperparameter
search (Section 2.4).

Usage:  python src/run_ablation.py
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (
    RESULT_DIR,
    SEED,
    SVM_KWARGS,
    banner,
    feature_matrix,
    load_all,
)

CONFIGURATIONS = [
    ("Color only", ("color",)),
    ("Texture only", ("texture",)),
    ("Shape only", ("shape",)),
    ("Color + Texture", ("color", "texture")),
    ("Color + Shape", ("color", "shape")),
    ("Texture + Shape", ("texture", "shape")),
    ("All (Color + Texture + Shape)", ("color", "texture", "shape")),
]

PCA_COMPONENTS = 60


def build(use_pca, n_features):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(
            ("pca", PCA(n_components=min(PCA_COMPONENTS, n_features), random_state=SEED))
        )
    steps.append(("clf", SVC(**SVM_KWARGS)))
    return Pipeline(steps)


def main():
    train, val, test, columns = load_all()
    y_train = train["label_id"].values
    y_test = test["label_id"].values

    banner("Table 5 - ablation over feature groups")
    rows = []
    for label, groups in CONFIGURATIONS:
        X_train = feature_matrix(train, columns, groups)
        X_test = feature_matrix(test, columns, groups)
        row = {"Feature Set": label, "#Features": X_train.shape[1]}

        for use_pca in (False, True):
            model = build(use_pca, X_train.shape[1])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            key = "Test Macro-F1 (PCA, %)" if use_pca else "Test Macro-F1 (no PCA, %)"
            row[key] = round(f1_score(y_test, pred, average="macro") * 100, 2)
            if not use_pca:
                row["Test Accuracy (%)"] = round(accuracy_score(y_test, pred) * 100, 2)

        rows.append(row)
        print(
            f"{label:32s} d={row['#Features']:3d}  "
            f"no-PCA {row['Test Macro-F1 (no PCA, %)']:6.2f}  "
            f"PCA {row['Test Macro-F1 (PCA, %)']:6.2f}"
        )

    table = pd.DataFrame(rows)[
        [
            "Feature Set",
            "#Features",
            "Test Macro-F1 (no PCA, %)",
            "Test Macro-F1 (PCA, %)",
            "Test Accuracy (%)",
        ]
    ]
    table.to_csv(RESULT_DIR / "table5_ablation.csv", index=False)
    print(f"\nwrote {RESULT_DIR / 'table5_ablation.csv'}")


if __name__ == "__main__":
    main()

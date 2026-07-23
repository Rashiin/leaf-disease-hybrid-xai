"""
Shared configuration and data loading for the leaf-disease hybrid-XAI pipeline.

Every script in src/ imports from here so that the preprocessing, the
hyperparameters and the train/validation/test partition are identical
across the baseline, ablation, fusion, stability and selective-prediction
experiments -- exactly as described in Section 2.7 of the paper.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"
SPLIT_DIR = ROOT / "data" / "splits"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = RESULT_DIR / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Fixed experimental configuration (Section 2.4)
# --------------------------------------------------------------------------
SEED = 42
SVM_KWARGS = dict(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    random_state=SEED,
)

GROUPS = ("color", "texture", "shape")
META_COLUMNS = ("path", "label", "label_id")

# Fusion weights reported in Table 3 / Figure 2(a), in (color, texture, shape)
# order.  See the note in run_fusion.py about why these are pinned.
PAPER_ALPHA = np.array([0.50761492, 0.17937671, 0.31300838])


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_group(group, split):
    """Read one feature-group CSV for one split, sorted by image path."""
    path = FEATURE_DIR / f"{group}_features_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The feature CSVs ship with this repository; "
            "if you removed them, regenerate them with notebooks/"
            "leaf_disease_pipeline.ipynb."
        )
    return pd.read_csv(path).sort_values("path").reset_index(drop=True)


def load_split(split):
    """
    Load one split and return (dataframe, column_index).

    The returned dataframe holds the 87 handcrafted features plus the three
    metadata columns.  column_index maps each feature-group name to its list
    of column names, so a script can select any subset without hard-coding
    dimensionalities.
    """
    frames = {g: _read_group(g, split) for g in GROUPS}

    reference = frames["color"]["path"]
    for g in GROUPS[1:]:
        if not (frames[g]["path"] == reference).all():
            raise ValueError(f"path mismatch between color and {g} for '{split}'")

    columns = {
        g: [c for c in frames[g].columns if c not in META_COLUMNS] for g in GROUPS
    }

    merged = frames["color"].copy()
    for g in GROUPS[1:]:
        for column in columns[g]:
            merged[column] = frames[g][column].values

    return merged, columns


def load_all():
    """Load train/validation/test together and check the column index agrees."""
    train, columns = load_split("train")
    val, columns_val = load_split("val")
    test, columns_test = load_split("test")

    if columns != columns_val or columns != columns_test:
        raise ValueError("feature columns differ between splits")

    return train, val, test, columns


def feature_matrix(frame, columns, groups=GROUPS):
    """Stack the requested feature groups into a single float array."""
    selected = []
    for g in groups:
        selected.extend(columns[g])
    return frame[selected].values.astype(float)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_svm(probability=False):
    """
    The classifier used everywhere in the paper: median imputation, z-score
    standardisation fitted on the training split, then an RBF-kernel SVM with
    the fixed hyperparameters of Section 2.4.

    `probability=True` adds Platt scaling.  It does not change `predict`, so
    the accuracy and Macro-F1 of Tables 4 and 6 are identical either way; the
    calibrated posteriors are needed only by run_selective.py.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=probability, **SVM_KWARGS)),
        ]
    )


def class_names(frame):
    """Ordered class names, indexed by label_id."""
    lookup = (
        frame[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("label_id")["label"]
    )
    return lookup.to_list()


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)

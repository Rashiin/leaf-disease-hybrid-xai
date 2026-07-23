"""
Attention-style convex expert fusion  ->  Table 3, Table 4 (fusion row),
the effective contribution of Section 3.4, Figure 2 and Figure 3.

A note on the fusion weights
----------------------------
The weights are selected by sampling 1000 Dirichlet(1,1,1) candidates on the
validation set.  As Section 3.4 reports, the objective has a broad
near-optimal plateau: 47 of 1326 grid points lie within 0.2 Macro-F1 points
of the optimum, spanning color 0.42-0.62, shape 0.18-0.42 and texture
0.10-0.30.  A random search therefore lands on a *different point of the
same plateau* depending on how the random stream is consumed, and the exact
triple is not reproducible across environments even at a fixed seed.

The paper's operating point is pinned in common.PAPER_ALPHA and used by
default so that the reported numbers regenerate exactly.  Pass --search to
run the Dirichlet search instead and see where it lands in your environment;
the effective-contribution ordering (color > texture > shape) is stable
either way, which is the claim the paper actually makes.

Usage:
    python src/run_fusion.py            # reproduce the reported numbers
    python src/run_fusion.py --search   # re-run the validation search
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import (
    FIGURE_DIR,
    GROUPS,
    PAPER_ALPHA,
    RESULT_DIR,
    SEED,
    banner,
    class_names,
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
    parser.add_argument(
        "--search",
        action="store_true",
        help="re-run the Dirichlet search instead of using the reported weights",
    )
    args = parser.parse_args()

    train, val, test, columns = load_all()
    names = class_names(train)
    y_train = train["label_id"].values
    y_val = val["label_id"].values
    y_test = test["label_id"].values

    banner("Training the three feature-group experts")
    posteriors_val, posteriors_test = [], []
    for group in GROUPS:
        model = make_svm(probability=True)
        model.fit(feature_matrix(train, columns, (group,)), y_train)
        posteriors_val.append(model.predict_proba(feature_matrix(val, columns, (group,))))
        posteriors_test.append(model.predict_proba(feature_matrix(test, columns, (group,))))
        print(f"  {group} expert trained")

    # ---- weights ----------------------------------------------------------
    if args.search:
        banner(f"Dirichlet search on the validation set ({N_TRIALS} candidates)")
        rng = np.random.default_rng(SEED)
        best_alpha, best_score = None, -np.inf
        for _ in range(N_TRIALS):
            candidate = rng.dirichlet([1.0, 1.0, 1.0])
            score = f1_score(
                y_val, fuse(candidate, posteriors_val).argmax(axis=1), average="macro"
            )
            if score > best_score:
                best_alpha, best_score = candidate, score
        alpha = best_alpha
        print(f"search optimum: validation Macro-F1 {best_score * 100:.2f}%")
    else:
        alpha = PAPER_ALPHA
        banner("Using the fusion weights reported in Table 3")

    print(f"alpha (color, texture, shape) = "
          f"{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}")
    pd.DataFrame(
        {"group": list(GROUPS), "weight": np.round(alpha, 3)}
    ).to_csv(RESULT_DIR / "table3_fusion_weights.csv", index=False)

    # ---- Table 4 fusion row ----------------------------------------------
    banner("Table 4 - fused model performance")
    rows = []
    for split_name, posteriors, y in (
        ("validation", posteriors_val, y_val),
        ("test", posteriors_test, y_test),
    ):
        pred = fuse(alpha, posteriors).argmax(axis=1)
        accuracy = accuracy_score(y, pred) * 100
        macro_f1 = f1_score(y, pred, average="macro") * 100
        rows.append(dict(split=split_name, accuracy=accuracy, macro_f1=macro_f1))
        print(f"{split_name:11s}: accuracy {accuracy:.2f}%   Macro-F1 {macro_f1:.2f}%")

    uniform = fuse(np.repeat(1 / 3, 3), posteriors_test).argmax(axis=1)
    print(f"uniform 1/3 control (test Macro-F1): "
          f"{f1_score(y_test, uniform, average='macro') * 100:.2f}%")
    pd.DataFrame(rows).to_csv(RESULT_DIR / "table4_fusion.csv", index=False)

    # ---- effective contribution (Section 3.4) ----------------------------
    banner("Section 3.4 - raw coefficients versus effective contribution")
    index = np.arange(len(y_test))
    effective = np.array(
        [a * P[index, y_test].mean() for a, P in zip(alpha, posteriors_test)]
    )
    shares = effective / effective.sum()
    sharpness = [P.max(axis=1).mean() for P in posteriors_test]

    summary = pd.DataFrame(
        {
            "group": list(GROUPS),
            "raw_coefficient": np.round(alpha, 3),
            "effective_contribution": np.round(effective, 3),
            "normalised_share_%": np.round(shares * 100, 1),
            "mean_max_posterior": np.round(sharpness, 2),
        }
    )
    print(summary.to_string(index=False))
    summary.to_csv(RESULT_DIR / "section34_effective_contribution.csv", index=False)

    # ---- Figure 2 ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.4), dpi=200)
    labels = [g.capitalize() for g in GROUPS]
    axes[0].bar(labels, alpha, color="#1f4e79", width=0.55)
    axes[0].set_ylabel("Coefficient (sum = 1)", fontsize=8)
    axes[0].set_title("(a) Raw fusion coefficient", fontsize=8.5)
    axes[1].bar(labels, shares, color="#c55a11", width=0.55)
    axes[1].set_ylabel("Normalized share", fontsize=8)
    axes[1].set_title("(b) Effective contribution", fontsize=8.5)
    for ax in axes:
        ax.set_ylim(0, 0.85)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure2_attribution.png", bbox_inches="tight")

    # ---- Figure 3: class-wise attention-weighted contribution -------------
    heat = np.zeros((len(names), len(GROUPS)))
    for class_id in range(len(names)):
        mask = y_test == class_id
        for j, P in enumerate(posteriors_test):
            heat[class_id, j] = alpha[j] * P[mask, class_id].mean()
    heat = heat / heat.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.4, 5.4), dpi=200)
    image = ax.imshow(heat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(image, ax=ax, label="Mean attention-weighted P(true class)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure3_class_wise.png", bbox_inches="tight")

    pd.DataFrame(heat, index=names, columns=labels).round(3).to_csv(
        RESULT_DIR / "figure3_class_wise.csv"
    )
    print(f"\nwrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

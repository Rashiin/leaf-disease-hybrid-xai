"""
Plateau sweep  ->  the near-equivalent-weighting analysis of Section 3.4.

Why this script exists
----------------------
Section 3.4 reports that the validation objective has a broad near-optimal
plateau, and that the *point* on that plateau which a random search happens
to reach decides whether the raw fusion coefficients agree with the ablation
ordering or contradict it.  The claim the paper makes is therefore not about
one triple of weights but about the whole plateau: across near-equivalent
weightings the raw coefficients recover the ablation ordering on roughly a
quarter of them, the effective contribution on roughly four fifths.

This script measures exactly that.  It samples `--candidates` Dirichlet
weightings, keeps those whose validation Macro-F1 lies within a tolerance of
the best candidate found, and for each retained weighting asks whether the
raw coefficients and the effective contribution reproduce the ordering
measured by the single-group ablation of Table 5.

The single-group ablation ordering is not hard-coded: it is recomputed here
from the same three experts, so the comparison is self-contained.

Usage:
    python src/run_plateau.py
    python src/run_plateau.py --candidates 4000 --tolerances 0.1 0.2 0.5
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import (
    GROUPS,
    RESULT_DIR,
    SEED,
    banner,
    feature_matrix,
    load_all,
    make_svm,
)


def fuse(alpha, posteriors):
    stacked = sum(a * P for a, P in zip(alpha, posteriors))
    return stacked / (stacked.sum(axis=1, keepdims=True) + 1e-12)


def ordering(values):
    """Group names ordered from largest to smallest value."""
    return tuple(GROUPS[i] for i in np.argsort(values)[::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=4000)
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="Macro-F1 points below the search optimum that still count as "
        "near-equivalent",
    )
    args = parser.parse_args()

    train, val, test, columns = load_all()
    y_train = train["label_id"].values
    y_val = val["label_id"].values
    y_test = test["label_id"].values

    banner("Training the three feature-group experts")
    posteriors_val, posteriors_test, single_group_f1 = [], [], []
    for group in GROUPS:
        model = make_svm(probability=True)
        model.fit(feature_matrix(train, columns, (group,)), y_train)
        P_val = model.predict_proba(feature_matrix(val, columns, (group,)))
        P_test = model.predict_proba(feature_matrix(test, columns, (group,)))
        posteriors_val.append(P_val)
        posteriors_test.append(P_test)
        score = f1_score(y_test, P_test.argmax(axis=1), average="macro") * 100
        single_group_f1.append(score)
        print(f"  {group:8s} expert: single-group test Macro-F1 {score:.2f}%")

    ablation_order = ordering(single_group_f1)
    print(f"\nablation ordering (Table 5): {' > '.join(ablation_order)}")

    # ---- sweep ------------------------------------------------------------
    banner(f"Sampling {args.candidates} Dirichlet(1,1,1) weightings")
    rng = np.random.default_rng(SEED)
    alphas = rng.dirichlet([1.0, 1.0, 1.0], size=args.candidates)

    val_scores = np.empty(args.candidates)
    raw_match = np.empty(args.candidates, dtype=bool)
    eff_match = np.empty(args.candidates, dtype=bool)

    index = np.arange(len(y_test))
    # E[P_k(y_true | x)] does not depend on alpha, so it is computed once and
    # the effective contribution of a candidate is alpha * this vector.
    mass_on_true = np.array([P[index, y_test].mean() for P in posteriors_test])

    for i, alpha in enumerate(alphas):
        val_scores[i] = (
            f1_score(y_val, fuse(alpha, posteriors_val).argmax(axis=1), average="macro")
            * 100
        )
        raw_match[i] = ordering(alpha) == ablation_order
        eff_match[i] = ordering(alpha * mass_on_true) == ablation_order
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.candidates} evaluated")

    best = val_scores.max()
    best_alpha = alphas[val_scores.argmax()]
    print(
        f"\nsearch optimum: validation Macro-F1 {best:.2f}% at "
        f"(color {best_alpha[0]:.3f}, texture {best_alpha[1]:.3f}, "
        f"shape {best_alpha[2]:.3f})"
    )

    # ---- report per tolerance --------------------------------------------
    banner("Section 3.4 - agreement with the ablation ordering on the plateau")
    rows = []
    for tol in sorted(args.tolerances):
        keep = val_scores >= best - tol
        n = int(keep.sum())
        raw_rate = raw_match[keep].mean() * 100
        eff_rate = eff_match[keep].mean() * 100
        rows.append(
            dict(
                tolerance_macro_f1_points=tol,
                n_retained=n,
                n_candidates=args.candidates,
                raw_coefficient_agreement_pct=round(raw_rate, 1),
                effective_contribution_agreement_pct=round(eff_rate, 1),
            )
        )
        print(
            f"tolerance {tol:.1f} pts: {n:4d} retained | "
            f"raw {raw_rate:5.1f}%  effective {eff_rate:5.1f}%"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "section34_plateau.csv", index=False)

    pd.DataFrame(
        {
            "alpha_color": alphas[:, 0],
            "alpha_texture": alphas[:, 1],
            "alpha_shape": alphas[:, 2],
            "val_macro_f1": val_scores,
            "raw_matches_ablation": raw_match,
            "effective_matches_ablation": eff_match,
        }
    ).to_csv(RESULT_DIR / "section34_plateau_candidates.csv", index=False)

    print(f"\nwrote {RESULT_DIR / 'section34_plateau.csv'}")
    print(
        "\nThe rates printed above are the ones quoted in Section 3.4. They are "
        "properties of the plateau, not of any single reported triple, which is "
        "why the agreement seen at one operating point should not be read as a "
        "property of the coefficients."
    )


if __name__ == "__main__":
    main()

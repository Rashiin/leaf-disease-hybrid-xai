#!/usr/bin/env python3
"""
reproduce_all.py — regenerates every number in

  "Evaluation shortcuts inflate the apparent value of temporal energy
   descriptors in speech arousal estimation"

from the released feature table.

Usage:
    python reproduce_all.py --features features_temporal_energy_plus_vq.csv

Outputs (written to ./results/):
    table1_leakage.csv      duplicate-leakage comparison
    table2_ablation.csv     all feature sets, corrected protocol
    table3_decomposition.csv
    table4_panel.csv        classifier panel, seed 0
    Fig2.pdf / Fig2.eps     SHAP beeswarm, held-out folds only
    shap_ranking.csv        mean |SHAP| per feature
    deployment.json         model size / support vectors
    console log of everything above

Protocol, exactly as described in the paper:
  * de-duplication on file basename BEFORE any split (2880 -> 1440)
  * standardisation fitted on training folds only
  * hyperparameter grid search on an inner 3-fold split of the training
    partition only
  * speaker-independent folds drawn from stated seeds, not GroupKFold
  * the whole speaker-independent evaluation repeated over 30 random
    actor partitions; mean +/- SD reported across partitions

Runtime is roughly two hours on two cores at the default 30 partitions.
Use --partitions 6 for a quick check; use --skip-shap to omit Figure 2.
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- features

TEMPORAL = ["d_rms_abs_mean", "d_rms_abs_std", "d_rms_abs_max",
            "rms_varwin_mean", "rms_varwin_std", "rms_varwin_max",
            "rms_slope", "rms_burstiness", "rms_segment_ratio_start_end"]
STATIC   = ["rms_mean", "rms_entropy"]
SPECTRAL = ["zcr_mean", "zcr_std", "spectral_flux_mean"]
PHONATORY = ["jitter_local", "shimmer_local", "hnr_mean"]
F0       = ["f0_mean", "f0_std"]
VQ       = PHONATORY + F0

SETS = {
    "Static":         STATIC,
    "Temporal":       TEMPORAL,
    "Fusion":         TEMPORAL + SPECTRAL,
    "F0 only":        F0,
    "Phonatory only": PHONATORY,
    "VQ":             VQ,
    "Temporal+VQ":    TEMPORAL + VQ,
    "Fusion+VQ":      TEMPORAL + SPECTRAL + VQ,
}

GRID = {"clf__C": [1, 10, 100], "clf__gamma": [1e-3, 1e-2, 1e-1]}
N_PARTITIONS = 30
SEEDS = list(range(N_PARTITIONS))


def pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, random_state=42))])


def fit_predict(Xtr, ytr, Xte, tuned):
    """Fit on the training partition, return probabilities for the test partition."""
    if tuned:
        gs = GridSearchCV(pipe(), GRID, scoring="f1", cv=3, n_jobs=-1)
        gs.fit(Xtr, ytr)
        m = gs.best_estimator_
    else:
        m = pipe().fit(Xtr, ytr)
    return m.predict(Xte), m.predict_proba(Xte)[:, 1]


# ------------------------------------------------------------------ splits

def actor_partition(actors, seed, n_folds=5):
    """Assign the 24 actors to n_folds groups from a stated seed.

    Drawn here rather than via GroupKFold: library implementations assign
    groups to folds by internal heuristics that differ between versions,
    so results conditioned on one such assignment are not reproducible
    across software environments.
    """
    uniq = np.sort(np.unique(actors))
    rng = np.random.RandomState(seed)
    order = rng.permutation(uniq)
    fold_of = {a: i % n_folds for i, a in enumerate(order)}
    return np.array([fold_of[a] for a in actors])


def eval_speaker_independent(df, cols, seed, tuned=True):
    X = df[cols].fillna(0).values
    y = df["label"].values
    folds = actor_partition(df["actor"].values, seed)
    f1s, aucs = [], []
    for k in range(folds.max() + 1):
        te = folds == k
        tr = ~te
        pred, proba = fit_predict(X[tr], y[tr], X[te], tuned)
        f1s.append(f1_score(y[te], pred))
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(f1s)), float(np.mean(aucs))


def eval_stratified(df, cols, tuned=True, seed=42):
    X = df[cols].fillna(0).values
    y = df["label"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s, aucs = [], []
    for tr, te in skf.split(X, y):
        pred, proba = fit_predict(X[tr], y[tr], X[te], tuned)
        f1s.append(f1_score(y[te], pred))
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(f1s)), float(np.mean(aucs))


# ------------------------------------------------------------------- tables

def table1_leakage(df_dup, df_dedup, out):
    """Untuned stratified 5-fold, duplicates kept vs removed."""
    rows = []
    for name in ["Static", "Temporal", "Fusion", "VQ", "Temporal+VQ", "Fusion+VQ"]:
        cols = SETS[name]
        f1d, aucd = eval_stratified(df_dup,   cols, tuned=False)
        f1c, aucc = eval_stratified(df_dedup, cols, tuned=False)
        rows.append(dict(feature_set=name,
                         dup_F1=round(f1d, 3),   dup_AUC=round(aucd, 3),
                         dedup_F1=round(f1c, 3), dedup_AUC=round(aucc, 3),
                         infl_F1=round(f1d - f1c, 3), infl_AUC=round(aucd - aucc, 3)))
        print(f"  {name:14s} dup {f1d:.3f}/{aucd:.3f}   dedup {f1c:.3f}/{aucc:.3f}"
              f"   inflation {f1d-f1c:+.3f}/{aucd-aucc:+.3f}")
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out, "table1_leakage.csv"), index=False)
    return t


def table2_ablation(df, out):
    """All feature sets, corrected protocol, N_PARTITIONS partitions."""
    rows, per_partition = [], {}
    for name, cols in SETS.items():
        sf1, sauc = eval_stratified(df, cols, tuned=True)
        scores = [eval_speaker_independent(df, cols, s) for s in SEEDS]
        per_partition[name] = scores
        f1s = np.array([a for a, _ in scores]); aucs = np.array([b for _, b in scores])
        rows.append(dict(feature_set=name, n=len(cols),
                         strat_F1=round(sf1, 3), strat_AUC=round(sauc, 3),
                         si_F1=round(f1s.mean(), 3),  si_F1_sd=round(f1s.std(), 3),
                         si_AUC=round(aucs.mean(), 3), si_AUC_sd=round(aucs.std(), 3)))
        print(f"  {name:15s} n={len(cols):2d}  strat {sf1:.3f}/{sauc:.3f}"
              f"   SI {f1s.mean():.3f}±{f1s.std():.3f} / {aucs.mean():.3f}±{aucs.std():.3f}")
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out, "table2_ablation.csv"), index=False)
    return t, per_partition


def significance(per_partition):
    """Paired comparison VQ vs Temporal+VQ across all partitions."""
    a = np.array(per_partition["VQ"]); b = np.array(per_partition["Temporal+VQ"])
    res = {}
    for j, metric in enumerate(["F1", "AUC"]):
        d = b[:, j] - a[:, j]
        t, pt = stats.ttest_rel(b[:, j], a[:, j])
        try:
            _, pw = stats.wilcoxon(b[:, j], a[:, j])
        except ValueError:
            pw = float("nan")
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
        res[metric] = dict(delta=round(float(d.mean()), 4),
                           p_ttest=float(pt), p_wilcoxon=float(pw),
                           ci_low=round(float(ci[0]), 4), ci_high=round(float(ci[1]), 4),
                           positive_partitions=int((d > 0).sum()), n=len(d))
        print(f"  Delta{metric} = {d.mean():+.4f}  paired-t p={pt:.4g}  "
              f"Wilcoxon p={pw:.4g}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"positive in {int((d>0).sum())}/{len(d)} partitions")
    return res


def table3_decomposition(df_dup, df_dedup, out):
    """Speaker-independent Delta(Temporal+VQ - VQ) under three protocols."""
    rows = []
    def delta(df, tuned):
        f1s, aucs = [], []
        for s in SEEDS:
            a = eval_speaker_independent(df, SETS["VQ"], s, tuned)
            b = eval_speaker_independent(df, SETS["Temporal+VQ"], s, tuned)
            f1s.append(b[0] - a[0]); aucs.append(b[1] - a[1])
        return float(np.mean(f1s)), float(np.mean(aucs))

    for label, d, tuned in [("Untuned, duplicates kept", df_dup,   False),
                            ("Untuned, de-duplicated",   df_dedup, False),
                            ("Tuned, de-duplicated",     df_dedup, True)]:
        f1, auc = delta(d, tuned)
        rows.append(dict(protocol=label, dF1=round(f1, 3), dAUC=round(auc, 3)))
        print(f"  {label:26s} dF1={f1:+.3f}  dAUC={auc:+.3f}")
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out, "table3_decomposition.csv"), index=False)
    return t


def table4_panel(df, out, seed=0):
    """Classifier panel on one shared speaker-independent partition."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    GENERIC = STATIC + SPECTRAL + ["rms_slope"]   # 6 conventional acoustic features
    configs = [
        ("Generic acoustic (6) + RBF-SVM", GENERIC,            lambda: SVC(kernel="rbf", probability=True, random_state=42)),
        ("VQ + k-NN",                      VQ,                 lambda: KNeighborsClassifier(n_neighbors=15)),
        ("VQ + MLP",                       VQ,                 lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=42)),
        ("Temporal only + RBF-SVM",        TEMPORAL,           lambda: SVC(kernel="rbf", probability=True, random_state=42)),
        ("VQ + logistic regression",       VQ,                 lambda: LogisticRegression(max_iter=2000)),
        ("VQ + random forest",             VQ,                 lambda: RandomForestClassifier(n_estimators=400, random_state=42)),
        ("VQ + RBF-SVM",                   VQ,                 lambda: SVC(kernel="rbf", probability=True, random_state=42)),
        ("Temporal+VQ + RBF-SVM",          TEMPORAL + VQ,      lambda: SVC(kernel="rbf", probability=True, random_state=42)),
    ]
    folds = actor_partition(df["actor"].values, seed)
    y = df["label"].values
    rows = []
    for name, cols, mk in configs:
        X = df[cols].fillna(0).values
        f1s, aucs = [], []
        for k in range(folds.max() + 1):
            te = folds == k; tr = ~te
            m = Pipeline([("sc", StandardScaler()), ("clf", mk())]).fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])[:, 1]
            f1s.append(f1_score(y[te], m.predict(X[te])))
            aucs.append(roc_auc_score(y[te], proba))
        rows.append(dict(method=name, F1=round(np.mean(f1s), 3), F1_sd=round(np.std(f1s), 3),
                         AUC=round(np.mean(aucs), 3), AUC_sd=round(np.std(aucs), 3)))
        print(f"  {name:32s} F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out, "table4_panel.csv"), index=False)
    return t


def deployment(df, out):
    """Serialised size and support-vector count of the released VQ model."""
    import pickle, tempfile
    res = {}
    for name in ["VQ", "Temporal+VQ"]:
        X = df[SETS[name]].fillna(0).values; y = df["label"].values
        gs = GridSearchCV(pipe(), GRID, scoring="f1", cv=3, n_jobs=-1).fit(X, y)
        m = gs.best_estimator_
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(m, f); path = f.name
        kb = os.path.getsize(path) / 1024
        os.remove(path)
        nsv = int(m.named_steps["clf"].support_vectors_.shape[0])
        res[name] = dict(size_kB=round(kb, 2), support_vectors=nsv,
                         best_params={k: v for k, v in gs.best_params_.items()})
        print(f"  {name:12s} {kb:.2f} kB, {nsv} support vectors, {gs.best_params_}")
    json.dump(res, open(os.path.join(out, "deployment.json"), "w"), indent=2)
    return res


def shap_attribution(df, out, seed=0, nsamples=100):
    """SHAP on held-out folds only, Temporal+VQ, pooled across folds."""
    import shap
    cols = SETS["Temporal+VQ"]
    X = df[cols].fillna(0).values; y = df["label"].values
    folds = actor_partition(df["actor"].values, seed)
    pooled, explained = [], []
    for k in range(folds.max() + 1):
        te = folds == k; tr = ~te
        gs = GridSearchCV(pipe(), GRID, scoring="f1", cv=3, n_jobs=-1).fit(X[tr], y[tr])
        m = gs.best_estimator_
        bg = shap.kmeans(X[tr], 25)                       # background: TRAINING fold only
        ex = shap.KernelExplainer(lambda z: m.predict_proba(z)[:, 1], bg)
        idx = np.random.RandomState(seed).choice(np.where(te)[0],
                                                 size=min(nsamples, te.sum()), replace=False)
        pooled.append(ex.shap_values(X[idx], nsamples=100))  # HELD-OUT samples only
        explained.append(X[idx])
    vals = np.vstack(pooled); Xexp = np.vstack(explained)
    rank = pd.DataFrame({"feature": cols,
                         "mean_abs_shap": np.abs(vals).mean(0)}).sort_values(
                         "mean_abs_shap", ascending=False).reset_index(drop=True)
    rank.to_csv(os.path.join(out, "shap_ranking.csv"), index=False)
    np.save(os.path.join(out, "shap_values.npy"), vals)
    np.save(os.path.join(out, "shap_explained_X.npy"), Xexp)
    print(rank.to_string(index=False))
    make_figure2(vals, Xexp, cols, out)
    return rank


# ------------------------------------------------------------------ Figure 2

DISPLAY_NAMES = {
    "hnr_mean": "HNR (mean)",
    "f0_mean": "$F_0$ (mean)",
    "f0_std": "$F_0$ (SD)",
    "jitter_local": "Jitter (local)",
    "shimmer_local": "Shimmer (local)",
    "d_rms_abs_mean": r"$|\Delta$RMS$|$ (mean)",
    "d_rms_abs_std": r"$|\Delta$RMS$|$ (SD)",
    "d_rms_abs_max": r"$|\Delta$RMS$|$ (max)",
    "rms_varwin_mean": "RMS rolling var. (mean)",
    "rms_varwin_std": "RMS rolling var. (SD)",
    "rms_varwin_max": "RMS rolling var. (max)",
    "rms_slope": "RMS slope",
    "rms_burstiness": "RMS burstiness",
    "rms_segment_ratio_start_end": "RMS start/end ratio",
}


def make_figure2(vals, Xexp, cols, out, width_mm=84.0, height_mm=78.0):
    """Beeswarm SHAP summary, Springer Nature single-column spec."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6.5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    order = np.argsort(np.abs(vals).mean(0))            # ascending: best at top
    cmap = LinearSegmentedColormap.from_list("vq", ["#2b6ca3", "#b8b8b8", "#c1443c"])

    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    rng = np.random.RandomState(0)
    for row, j in enumerate(order):
        s = vals[:, j]
        v = Xexp[:, j].astype(float)
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        c = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        # vertical jitter proportional to local density
        hist, edges = np.histogram(s, bins=24)
        dens = hist[np.clip(np.digitize(s, edges) - 1, 0, len(hist) - 1)]
        spread = 0.34 * dens / (dens.max() + 1e-12)
        yj = row + rng.uniform(-1, 1, size=len(s)) * spread
        ax.scatter(s, yj, c=c, cmap=cmap, s=1.6, linewidths=0, alpha=0.85,
                   rasterized=False, vmin=0, vmax=1)

    ax.axvline(0, color="0.55", lw=0.5, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY_NAMES.get(cols[j], cols[j]) for j in order])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("SHAP value (impact on predicted high-arousal probability)", fontsize=6.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6)

    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.032, aspect=26)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", fontsize=6.5, labelpad=-6)
    cb.ax.tick_params(labelsize=6, length=0); cb.outline.set_visible(False)

    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(out, "Fig2.pdf"), dpi=1200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "Fig2.eps"), format="eps", dpi=1200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "Fig2.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Figure 2 written to", out)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_temporal_energy_plus_vq.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--partitions", type=int, default=N_PARTITIONS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global SEEDS
    SEEDS = list(range(a.partitions))

    df_dup = pd.read_csv(a.features)
    df_dup["base"] = df_dup["path"].apply(lambda p: os.path.basename(str(p)))
    df = df_dup.drop_duplicates("base").reset_index(drop=True)

    print(f"\nRaw listing:      {len(df_dup)} rows")
    print(f"De-duplicated:    {len(df)} utterances "
          f"({int(df.label.sum())} high-arousal / {int((df.label==0).sum())} control, "
          f"{df.actor.nunique()} actors)\n")

    print("== Table 1: duplicate leakage (untuned, stratified) ==")
    table1_leakage(df_dup, df, a.out)

    print(f"\n== Table 2: ablation, corrected protocol, {len(SEEDS)} partitions ==")
    _, per_partition = table2_ablation(df, a.out)

    print("\n== Significance: VQ vs Temporal+VQ, paired across partitions ==")
    sig = significance(per_partition)
    json.dump(sig, open(os.path.join(a.out, "significance.json"), "w"), indent=2)

    print("\n== Table 3: decomposition ==")
    table3_decomposition(df_dup, df, a.out)

    print("\n== Table 4: classifier panel (seed 0) ==")
    table4_panel(df, a.out)

    print("\n== Deployment profile ==")
    deployment(df, a.out)

    if not a.skip_shap:
        print("\n== SHAP attribution (held-out folds only) ==")
        shap_attribution(df, a.out)

    print(f"\nAll outputs written to {a.out}/")


if __name__ == "__main__":
    main()

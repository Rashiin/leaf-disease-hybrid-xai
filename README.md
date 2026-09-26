# Learned Fusion Weights Are Not Feature Importance

Code, features and results accompanying the manuscript *"Learned Fusion Weights Are Not Feature Importance in
Convex Expert Combination for Leaf Disease Classification"* by Rashin Gholijani Farahani, Azam Bastanfard and
Javad Mohammadzadeh (submitted to the Journal of Artificial Intelligence and Data Mining, JAIDM).

Three feature-group experts (color, texture, shape; 87 handcrafted features) are trained as RBF-kernel SVMs on a
15-class PlantVillage release (20,638 images) and combined by convex weights. Because the experts differ in
posterior sharpness, the raw fusion weights are not valid feature-importance scores; the *effective
contribution* (weight x posterior mass on the correct class) is. The repository reproduces every table and
figure of the manuscript and its supplement.

## What is in this repository

```
data/features/    87 handcrafted features per image: color (57) / texture (22) / shape (8), train/val/test
data/splits/      the exact stratified 70/15/15 partition used in the paper
notebooks/        the original end-to-end notebook, including feature extraction from the raw images
                  (cell 17 defines the texture features used in the CSVs; cell 9 is an earlier 4-property version)
src/              one script per experiment (see below)
results/          outputs written by the scripts
extract_background_control.py   background-masked colour features (needs the raw images)
```

Because the extracted features ship with the repository, every handcrafted-feature experiment can be
regenerated without downloading the PlantVillage images.

## Quick start

```bash
git clone https://github.com/Rashiin/leaf-disease-hybrid-xai.git
cd leaf-disease-hybrid-xai
pip install -r requirements.txt

python src/run_baseline.py            # Table 1 (baseline), Supplementary Table S1, Figure 2
python src/run_ablation.py            # Table 2, Supplementary Table S4
python src/run_fusion.py              # Table 3, Figure 1, Section 3.3
python src/run_plateau.py             # Section 3.3 plateau analysis
python src/run_stability.py           # ten re-splits: Section 3.3, Supplementary Table S2
python src/run_calibration_check.py   # Table 4 (Section 3.4)
python src/run_selective.py           # Table 6, Supplementary Figure S2 (Section 3.6)
python src/run_controls_fit.py paper 0 1 2 3 4 5 6 7 8 9   # Section 3.5 (slow: ~2-3 min per split)
python src/run_controls_analysis.py   # Table 5: label-free / class-balanced effective contribution,
                                      # leave-one-group-out and group-Shapley references,
                                      # NLL / Brier / logistic-stacking fusion, McNemar and Wilcoxon tests
```

The deep experiments read the raw PlantVillage images and need PyTorch (see `HOWTO_run_deep_moe.md`):

```bash
python src/run_deep_baseline.py       # MobileNetV2 reference: Table 1, McNemar test (Section 3.1)
python src/run_timing.py              # per-image latency, Table 1 and Supplementary S7
python src/run_deep_moe.py            # mixture of deep experts: Table 7, Supplementary Table S5 (Section 3.7)
```

Library versions of the deep runs are recorded in `results/deep_moe_environment.json`.

Script and output file names predate the final table numbering (for example `table8_stability.csv` holds
Supplementary Table S2); the mapping above is authoritative.

## Experimental configuration

| Setting | Value |
| --- | --- |
| Classifier | `SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")` |
| Preprocessing | median imputation, then z-score standardisation fitted on train |
| Partition | stratified 70/15/15 (14,446 / 3,096 / 3,096) |
| Fusion search | 1,000 Dirichlet(1,1,1) candidates, seed 42, validation Macro-F1 |
| Primary metric | Macro-F1 |

## Reproduction notes

These are the tolerances we observed when re-running the pipeline in a
different environment from the one used for the manuscript. They are stated
here so that a reader who gets a slightly different last digit knows whether
it matters.

**Reproduces exactly.** The baseline (validation 97.48/97.22, test
97.42/97.12), every cell of the per-class Supplementary Table S1, the effective
contributions of Section 3.3 (color 0.469 / 70.6%, texture 0.105 / 15.8%,
shape 0.091 / 13.6%), the expert posterior sharpness (0.94 / 0.68 / 0.42) and
all of Table 6 including the deferral analysis.

**Reproduces to within ~0.05 points.** Two ablation cells (texture-only,
and Color+Texture under PCA) and the fused model's accuracy and Macro-F1
moved by one or two test images out of 3,096 between scikit-learn versions.
The fused figures in Table 1 come from the authors' run; a current
scikit-learn gives 96.71 / 96.13 instead of 96.67 / 96.07, because Platt
scaling fits its calibration by internal cross-validation and is sensitive to
the library version. The predictions of the *uncalibrated* classifier — and
therefore Tables 1 (baseline) and 2 and Supplementary Table S1 — are unaffected.

**Depends on the random stream, by design.** The Dirichlet search over the
simplex has a broad near-optimal plateau: as Section 3.3 reports, 47 of 1,326
grid points lie within 0.2 Macro-F1 points of the optimum. A search therefore
lands on a different point of the same plateau depending on how the random
stream is consumed, and the exact weight triple is not portable across
environments. `run_fusion.py` uses the reported weights by default and
accepts `--search` to re-run the search. The claim the paper makes is about
the *effective contribution ordering* (color > texture > shape), which is
stable across restarts, grid search and re-splits, not about the exact
coefficients.

**`run_stability.py` reproduces the protocol, not the partitions.**
"Stratified 70/15/15 with seed *s*" does not identify a unique partition
across implementations, so the per-split rows of Supplementary Table S2 will differ. The
conclusions it supports — that the single reported split is not an outlier,
that the optimized fusion beats the uniform control, and that the effective
contribution recovers the ablation ordering far more reliably than the raw
coefficients — reproduce.

**The two attribution counts move by one split between environments.**
Section 3.3 reports that the raw coefficients recover the ablation ordering
in 3 of the 10 re-splits and the effective contribution in 8 of 10, with mean
Spearman correlations of 0.65 and 0.90 against the single-group Macro-F1
scores. Those figures come from the manuscript's own environment. Re-running
`run_calibration_check.py` on the pinned stack of `requirements.txt` gives 4
of 10 and 10 of 10, with correlations of 0.60 and 1.00. The cause is the
plateau described in the previous note: the search lands on a different point
of it, and near the edges of the plateau the *ordering* of the raw
coefficients flips. We report the manuscript's figures because they are the
ones the paper was written from, and because they are the more conservative
of the two -- the newer stack makes the effective contribution look better,
not worse. Expect these two counts to vary by roughly one split; the gap
between the two attributions does not.

**Reproduces exactly, on the pinned stack.** `run_plateau.py` returns 183 of
4,000 candidates within 0.2 Macro-F1 points of the search optimum, with the
raw coefficients recovering the ablation ordering on 24.6% of them and the
effective contribution on 81.4% (Section 3.3 quotes 25% and 81%); the rates
at tolerances of 0.1 and 0.5 points are 18.4/84.2% and 37.3/64.7% against the
quoted 18/84% and 37/65%. `run_calibration_check.py` returns the fitted
temperatures of Table 4 (color 0.804 +/- 0.021, texture 0.906 +/- 0.011,
shape 0.939 +/- 0.017), the same calibration errors and posterior sharpness
to three decimal places, and the same three-thousandth closing of the
sharpness gap (0.523 -> 0.520).

## Data

The images are the Kaggle release `emmarex/plantdisease` of PlantVillage (15 classes of pepper, potato and
tomato). Curation removed the aggregate `PlantVillage` directory that duplicates the class folders, leaving
20,638 images. Searching for identical feature vectors finds 14 duplicate pairs (no conflicting labels); five
test images have a copy in the training or validation split, and removing them changes the baseline test
Macro-F1 by less than 0.01 points.

## Scope

All results are obtained under controlled imaging conditions. Cross-dataset transfer and field acquisition are
outside the scope of these experiments. Because color supplies about 70% of the effective contribution, the
pipeline is a priori most exposed to illumination, white balance, compression and background differences;
`extract_background_control.py` recomputes the colour features on the leaf and on the background separately.

## License

MIT, see [LICENSE](LICENSE). The PlantVillage images are distributed under their own terms by their original
authors.

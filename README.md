# Lightweight Explainable Leaf Disease Classification Using Hybrid Features

Code and features accompanying the manuscript *"Lightweight Explainable Leaf
Disease Classification Using Hybrid Features"* (Journal of Artificial
Intelligence and Data Mining, JAIDM), by Rashin Gholijani Farahani, Azam
Bastanfard and Javad Mohammadzadeh.

An 87-dimensional handcrafted representation (color, texture, shape) with a
single RBF-kernel SVM reaches **97.42% accuracy / 97.12% Macro-F1** on a
curated 15-class PlantVillage subset, with no deep feature extractor, no GPU
and no lesion-segmentation stage. Three feature-group experts fused through
validation-optimized convex weights turn the classifier into a source of
quantitative, decision-level explanation, and a validation-calibrated reject
option raises accuracy to **99.53% at 90% coverage** while deferring 83.8% of
all errors.

## What is in this repository

```
data/features/    87 handcrafted features per image, split into
                  color (57) / texture (22) / shape (8) for train/val/test
data/splits/      the exact stratified 70/15/15 partition used in the paper
notebooks/        the original end-to-end notebook, including the
                  feature-extraction code that reads the raw images
src/              one script per experiment (see below)
results/          outputs written by the scripts; ablation_results.csv is
                  the authors' original ablation run
```

Because the extracted features ship with the repository, **every table and
figure can be regenerated without downloading the PlantVillage images.** The
raw images are only needed if you want to re-extract features from scratch.

## Quick start

```bash
git clone https://github.com/Rashiin/leaf-disease-hybrid-xai.git
cd leaf-disease-hybrid-xai
pip install -r requirements.txt

python src/run_baseline.py     # Table 4 (baseline), Table 6, Figure 1
python src/run_ablation.py     # Table 5
python src/run_fusion.py       # Table 3, Table 4 (fusion), Section 3.4, Figures 2-3
python src/run_selective.py    # Table 9, Figure 4  (Section 3.9)
python src/run_stability.py    # Table 8            (Section 3.8)
python src/run_plateau.py      # Section 3.4 plateau sweep
python src/run_calibration_check.py   # Table 10           (Section 3.10)
```

The first four scripts take a few minutes each on a single CPU core.
`run_plateau.py` trains the three experts once and then evaluates 4,000
Dirichlet weightings, so it finishes in a few minutes too. `run_stability.py`
and `run_calibration_check.py` refit the experts on each of ten re-splits and
take considerably longer; both accept `--splits 3` for a quick check.

## Experimental configuration

Fixed a priori and identical across every experiment (Section 2.4 of the
paper), so that comparisons between feature subsets reflect the information
content of those subsets rather than a per-subset hyperparameter search:

| Setting | Value |
| --- | --- |
| Classifier | `SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")` |
| Preprocessing | median imputation, then z-score standardisation fitted on train |
| Partition | stratified 70/15/15 (14,446 / 3,096 / 3,096) |
| Seed | 42 |
| Primary metric | Macro-F1 |

## Reproduction notes

These are the tolerances we observed when re-running the pipeline in a
different environment from the one used for the manuscript. They are stated
here so that a reader who gets a slightly different last digit knows whether
it matters.

**Reproduces exactly.** The baseline (validation 97.48/97.22, test
97.42/97.12), every cell of the per-class Table 6, the effective
contributions of Section 3.4 (color 0.469 / 70.6%, texture 0.105 / 15.8%,
shape 0.091 / 13.6%), the expert posterior sharpness (0.94 / 0.68 / 0.42) and
all of Table 9 including the deferral analysis.

**Reproduces to within ~0.05 points.** Two ablation cells (texture-only,
and Color+Texture under PCA) and the fused model's accuracy and Macro-F1
moved by one or two test images out of 3,096 between scikit-learn versions.
The fused figures in Table 4 come from the authors' run; a current
scikit-learn gives 96.71 / 96.13 instead of 96.67 / 96.07, because Platt
scaling fits its calibration by internal cross-validation and is sensitive to
the library version. The predictions of the *uncalibrated* classifier — and
therefore Tables 4 (baseline), 5 and 6 — are unaffected.

**Depends on the random stream, by design.** The Dirichlet search over the
simplex has a broad near-optimal plateau: as Section 3.4 reports, 47 of 1,326
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
across implementations, so the per-split rows of Table 8 will differ. The
conclusions it supports — that the single reported split is not an outlier,
that the optimized fusion beats the uniform control, and that the effective
contribution recovers the ablation ordering far more reliably than the raw
coefficients — reproduce.

**The two attribution counts move by one split between environments.**
Section 3.8 reports that the raw coefficients recover the ablation ordering
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
effective contribution on 81.4% (Section 3.4 quotes 25% and 81%); the rates
at tolerances of 0.1 and 0.5 points are 18.4/84.2% and 37.3/64.7% against the
quoted 18/84% and 37/65%. `run_calibration_check.py` returns the fitted
temperatures of Table 10 (color 0.804 +/- 0.021, texture 0.906 +/- 0.011,
shape 0.939 +/- 0.017), the same calibration errors and posterior sharpness
to three decimal places, and the same three-thousandth closing of the
sharpness gap (0.523 -> 0.520).

## Data

The images are the Kaggle release `emmarex/plantdisease` of PlantVillage.
Curation removed the aggregate `PlantVillage` directory that duplicates
images already present in the class folders and restricted the label set to
the fifteen intended crop-disease categories, leaving 20,638 images. No
image-level de-duplication or label-noise correction was applied, so that the
benchmark stays comparable with prior PlantVillage studies.

## Scope

All results are obtained under controlled imaging conditions. Cross-dataset
transfer, field acquisition and robustness to illumination, white balance and
compression are explicitly outside the scope of these experiments and are
discussed as future work in the paper. Because color supplies about 70% of
the effective contribution, the pipeline is a priori most exposed to exactly
those acquisition-side factors.

## Citation

```bibtex
@article{gholijanifarahani_leafdisease_jaidm,
  title   = {Lightweight Explainable Leaf Disease Classification Using Hybrid Features},
  author  = {Gholijani Farahani, Rashin and Bastanfard, Azam and Mohammadzadeh, Javad},
  journal = {Journal of Artificial Intelligence and Data Mining},
  note    = {In press}
}
```

## License

MIT, see [LICENSE](LICENSE). The PlantVillage images are distributed under
their own terms by their original authors.

# Corrections to arXiv v1

This document records defects found in the original implementation of this project, how they were found, and what changed as a result.

**It is kept in the repository so that anyone who read or cited arXiv v1 can see exactly what was wrong.**

The original notebook is preserved unmodified in `legacy/`.

---

## Summary for anyone who cited v1

| v1 claim | Status |
|---|---|
| F1 = 0.805, AUC = 0.861 (Temporal+VQ, stratified) | **Do not cite.** Produced by a leaky split; the de-duplicated value is 0.779 / 0.828 |
| Temporal-energy descriptors add +0.028 AUC over voice quality | **Overstated ~4×.** The corrected value is +0.009 AUC |
| SHAP identifies ΔRMS, rolling variance and burstiness as the main contributors | **Wrong.** SHAP was computed on training data for the wrong feature set. Recomputed correctly, HNR and F0 rank first and rolling variance contributes almost nothing |
| Feature-extraction latency figure | **Incomplete.** Only the RMS path was profiled; voice-quality extraction was never timed |
| Audio was peak-normalised | **Not performed.** No amplitude normalisation was applied |

The temporal-energy descriptors do still contribute — the gain is small but positive in every partition tested, 95% CI on ΔAUC = [+0.006, +0.011]. What the correction removes is roughly three quarters of the originally reported effect.

---

## How the defects were found

The manuscript's Methods section was compared line by line against the code that produced its results. Four claims did not match what the code executed; a fifth was found in the same pass.

## Defect 1 — de-duplication claimed but never performed

**Claimed:** *"a single unique file list was created based on file names. The resulting class distribution was 768 utterances (53.3%) ... 672 (46.7%)."*

**Actual:** the file list was built with

```python
for wp in glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True):
```

with no de-duplication anywhere. The distributed RAVDESS archive mirrors its contents, so this returns **2880 files for 1440 unique utterances**. Under `StratifiedKFold(shuffle=True)` the exact twin of every test utterance sat in the training partition.

**Decisive evidence:** re-running the original code *with* duplicates reproduces the old headline exactly — **F1 = 0.805, AUC = 0.861**. The same code on the de-duplicated set gives **0.779 / 0.828**. The published stratified column came from the leaky run.

Speaker-independent (`GroupKFold` by actor) results were *not* affected by this particular leak, because an utterance and its twin share an actor and therefore land in the same fold.

## Defect 2 — grid search claimed but never run

**Claimed:** *"Hyperparameters are tuned by grid search on training folds with C ∈ {1, 10, 100} and γ ∈ {0.001, 0.01, 0.1}."*

**Actual:** `GridSearchCV` does not appear anywhere in the notebook. Every model was

```python
SVC(kernel="rbf", probability=True, random_state=42)
```

at library defaults (`C=1.0`, `gamma='scale'`). Tuning changes the *relative ordering* of feature sets, so this was not a cosmetic omission.

## Defect 3 — SHAP computed on training data, on the wrong feature set

**Claimed:** *"attributions are computed on held-out test folds only, so explanations reflect generalization rather than memorized patterns."*

**Actual:**

```python
pipe.fit(X, y)              # fitted on the FULL dataset
bg = shap.kmeans(X, 30)     # background from the FULL dataset
X_sample = X[:200]          # explained samples are TRAINING data
shap_vals = explainer.shap_values(X_sample, nsamples=200)
```

Additionally the analysis ran with `feat_set = "fusion"`, so the published attribution figure described the **Fusion** configuration, not the **Temporal+VQ** configuration the paper proposed.

Recomputed correctly — model fitted per fold, background from that fold's training partition, attributions on held-out samples only — the ranking changes: **harmonics-to-noise ratio and F0 come first**, above every energy descriptor.

## Defect 4 — deployment profile measured only part of the pipeline

The reported feature-extraction time profiled `extract_features_full()`, which covers the RMS-contour and spectral descriptors. The Parselmouth voice-quality extraction (jitter, shimmer, HNR, F0) required by the proposed configuration was never timed. The corrected paper therefore reports model size and inference latency — which are independent of the audio front end — and deliberately reports no end-to-end extraction budget.

## Defect 5 — amplitude normalisation claimed but not applied

The Methods stated the audio was *"peak-normalized"*. The code calls `librosa.load(path, sr=16000)` only. The corrected text states plainly that no amplitude normalisation is applied, and notes the consequence: the energy descriptors encode the absolute recording scale of the corpus.

---

## What changed scientifically

Under the corrected protocol the paper's central claim shrinks by about a factor of four. It does not disappear.

| | v1 (shortcuts, one split) | Corrected (repeated partitions) |
|---|---|---|
| VQ, speaker-independent | 0.741 / 0.781 | 0.764 / 0.802 |
| Temporal+VQ, speaker-independent | 0.769 / 0.815 | 0.769 / 0.811 |
| **Gain from 9 temporal features** | **+0.028 / +0.034** | **+0.005 / +0.009** |
| | | *(p = 0.021, p = 0.0007; 6/6 partitions positive)* |

Decomposing where the inflation came from, speaker-independent, six partitions:

| Protocol | ΔF1 | ΔAUC |
|---|---|---|
| Untuned, duplicates kept | +0.020 | +0.028 |
| Untuned, de-duplicated | +0.012 | +0.025 |
| **Tuned, de-duplicated** | **+0.005** | **+0.009** |

Note carefully: for the **speaker-independent** column the dominant factor is defect 2 (no grid search), not defect 1 (duplicates). An utterance and its twin share an actor, so they land in the same fold and cannot leak across the split. Duplicate leakage dominates instead in the **stratified** column. An earlier draft of this correction attributed the whole effect to leakage; that was imprecise and has been fixed in both the paper and this document.

A separate methodological finding emerged during the correction. A **single** speaker-independent partition cannot resolve an effect of this size on a 24-speaker corpus: fold-to-fold SD is 0.03–0.05 AUC, and our single-partition runs produced p-values anywhere between 0.11 and 0.70 for the same comparison, depending only on which partition was drawn — and in one case on which scikit-learn version assigned the folds. An intermediate draft of this correction wrongly concluded "no effect (p = 0.94)" from one such partition. The repeated design fixed that. `reproduce_all.py` now draws partitions from stated seeds rather than relying on `GroupKFold`.

---

## Files superseded

Everything in `legacy/` corresponds to the defective analysis and should not be used:

- `speech_stress_temporal_energy.ipynb` — original notebook (defects 1–5)
- `fig_ablation_*.png` — ablation bars from the leaky, untuned runs
- `fig_shap_summary_fusion.png`, `fig_shap_dependence_*.png` — SHAP on training data, Fusion feature set

`fig_rms_trace_high.png` and `fig_rms_trace_low.png` are simple illustrations of RMS contours and remain factually accurate; they are kept in `legacy/` only because they belong to the original figure set.

---

## Reproducing

```bash
python reproduce_all.py
```

regenerates every table and figure in the corrected paper, including the leakage comparison in Table 1, from a clean checkout.

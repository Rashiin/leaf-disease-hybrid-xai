"""
Does the attribution correction survive when the experts are deep?
==================================================================

Section 3.4 of the paper shows that the raw coefficients of a convex expert
fusion are not feature-importance scores, because the three handcrafted
experts differ in posterior sharpness, and proposes the *effective
contribution* as the correction.  A referee can reasonably ask whether this
is a peculiarity of RBF-SVMs over 87 handcrafted dimensions.

This script answers that question.  It builds the same mixture out of
*deep* experts: one frozen ImageNet MobileNetV2 backbone applied to three
input streams that isolate the same three cues, a linear head per stream,
and the identical Dirichlet fusion protocol of Section 2.5.  It then repeats
the whole Section 3.4 / 3.8 / 3.10 analysis on that mixture:

  * single-stream Macro-F1                  -> the ablation ground truth
  * raw convex coefficients                 -> the attribution under test
  * effective contribution                  -> the proposed correction
  * posterior sharpness and entropy         -> the mechanism
  * per-expert temperature scaling          -> the calibration control
  * repetition over several stratified re-splits

The three streams
-----------------
color    the image reduced to a GRID x GRID colour layout and resized back,
         which destroys texture and fine shape while preserving chromatic
         composition;
texture  greyscale minus a Gaussian blur of itself (a high-pass), which
         removes colour and low-frequency shape and keeps micro-texture;
shape    the binary leaf mask of Section 2.2 (identical HSV thresholds and
         morphology), which keeps only the silhouette.

Because the backbone is frozen, the 1280-d embedding of every image under
every stream is computed once and cached; every split, head, fusion search
and control after that costs seconds.

Usage
-----
    python src/run_deep_moe.py --data-root /path/to/PlantVillage

--data-root is the directory holding the 15 class sub-directories.  The
paths in data/splits/*.csv are absolute paths from the machine the study was
run on and are rebased onto --data-root automatically.

    --splits N      number of extra stratified re-splits (default 9, so ten
                    partitions in total, matching Section 3.8)
    --limit N       use only ~N images per class directory (smoke test)
    --recompute     ignore the embedding cache

Requires:  pip install torch torchvision opencv-python
Outputs:   results/deep_moe_summary.csv
           results/deep_moe_sweep.csv
           results/deep_moe_per_split.csv
           results/deep_moe_calibration.csv
           results/deep_moe_environment.json
           results/cache/deep_moe_embeddings_<stream>.npz
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from common import RESULT_DIR, SPLIT_DIR, banner

SEED = 42
IMAGE_SIZE = 224
STREAMS = ("color", "texture", "shape")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CACHE_DIR = RESULT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
def stream_color(bgr, grid=16):
    """Chromatic composition only: down to grid x grid and back up."""
    small = cv2.resize(bgr, (grid, grid), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def stream_texture(bgr):
    """Micro-texture only: greyscale high-pass, colour and low frequencies gone."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    high = gray - low
    high = high - high.min()
    peak = high.max()
    high = high / peak * 255.0 if peak > 1e-6 else np.zeros_like(high)
    out = cv2.resize(high.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def stream_shape(bgr):
    """Silhouette only: the HSV leaf mask of Section 2.2, largest component."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    out = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class ThreeStreamDataset(Dataset):
    """Decodes each image once and returns all three streams."""

    def __init__(self, files, grid):
        self.files = list(files)
        self.grid = grid

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        bgr = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.files[i])
        if bgr.shape[0] != 256 or bgr.shape[1] != 256:
            bgr = cv2.resize(bgr, (256, 256))
        return (
            to_tensor(stream_color(bgr, self.grid)),
            to_tensor(stream_texture(bgr)),
            to_tensor(stream_shape(bgr)),
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def rebase(stored, data_root):
    stored = str(stored).replace("\\", "/")
    marker = "PlantVillage/"
    tail = stored.split(marker)[-1] if marker in stored else Path(stored).name
    return Path(data_root) / tail


def load_pool(data_root, limit=None):
    """Every image in the paper's partition, with its split membership."""
    frames = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{split}.csv").sort_values("path")
        frame["split"] = split
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True)
    pool["file"] = [rebase(p, data_root) for p in pool["path"]]
    if limit:
        # Sample inside every (split, class) cell, so the paper partition keeps a
        # non-empty validation and test set in smoke-test mode.
        per_cell = max(2, limit // (3 * pool["label_id"].nunique()))
        pool = (
            pool.groupby(["split", "label_id"], group_keys=False)
            .head(per_cell)
            .reset_index(drop=True)
        )
    missing = [f for f in pool["file"] if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(pool)} images not found under {data_root}. "
            f"First missing: {missing[0]}"
        )
    return pool.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def build_backbone(device):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()          # 1280-d pooled features
    return model.eval().to(device)


def compute_embeddings(pool, device, grid, batch_size, workers, recompute):
    caches = {s: CACHE_DIR / f"deep_moe_embeddings_{s}.npz" for s in STREAMS}
    key = np.array([str(p) for p in pool["file"]])
    if not recompute and all(c.exists() for c in caches.values()):
        loaded = {}
        ok = True
        for s, c in caches.items():
            z = np.load(c, allow_pickle=True)
            if len(z["paths"]) != len(key) or not (z["paths"] == key).all():
                ok = False
                break
            loaded[s] = z["X"]
        if ok:
            print("embeddings loaded from cache")
            return loaded
        print("cache does not match this image list; recomputing")

    backbone = build_backbone(device)
    loader = DataLoader(
        ThreeStreamDataset(pool["file"], grid),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    out = {s: [] for s in STREAMS}
    start = time.perf_counter()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            for s, tensor in zip(STREAMS, batch):
                out[s].append(backbone(tensor.to(device)).cpu().numpy())
            if bi % 20 == 0:
                done = bi * batch_size
                rate = done / (time.perf_counter() - start)
                print(f"  {done:6d}/{len(pool)} images  ({rate:.0f} img/s)", flush=True)
    embeddings = {s: np.concatenate(v).astype(np.float32) for s, v in out.items()}
    for s, c in caches.items():
        np.savez_compressed(c, X=embeddings[s], paths=key)
    print(f"embeddings computed in {(time.perf_counter() - start) / 60:.1f} min")
    return embeddings


# ---------------------------------------------------------------------------
# Experts, fusion, attribution  (protocol identical to Sections 2.5 and 3.4)
# ---------------------------------------------------------------------------
def fit_expert(Xtr, ytr, Xrest):
    """
    One linear head on a frozen-backbone embedding.

    Everything is cast to float64 first: on some macOS/Accelerate builds the
    float32 matmul path emits spurious overflow warnings inside scikit-learn,
    and float64 both silences them and removes any doubt about the numbers.
    """
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xrest = [np.asarray(X, dtype=np.float64) for X in Xrest]
    scaler = StandardScaler().fit(Xtr)
    head = LogisticRegression(
        max_iter=1000, tol=1e-3, C=1.0, random_state=SEED
    ).fit(scaler.transform(Xtr), ytr)
    out = [head.predict_proba(scaler.transform(X)) for X in Xrest]
    for p in out:
        if not np.isfinite(p).all():
            raise FloatingPointError("non-finite posterior from a linear head")
    return out


def dirichlet_search(probs_val, y_val, n_trials=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    best_alpha, best_score = None, -np.inf
    for _ in range(n_trials):
        alpha = rng.dirichlet(np.ones(len(probs_val)))
        fused = sum(a * p for a, p in zip(alpha, probs_val))
        score = f1_score(y_val, fused.argmax(1), average="macro")
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha, best_score * 100


def temperature_fit(prob, y, grid=np.linspace(0.30, 5.00, 236)):
    """One temperature per expert, fitted on validation NLL of log-probabilities."""
    logits = np.log(np.clip(prob, 1e-12, None))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        scaled = logits / t
        scaled = scaled - scaled.max(1, keepdims=True)
        p = np.exp(scaled)
        p /= p.sum(1, keepdims=True)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(prob, t):
    logits = np.log(np.clip(prob, 1e-12, None)) / t
    logits = logits - logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def ece(prob, y, bins=15):
    conf = prob.max(1)
    correct = (prob.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def spearman(a, b):
    """Spearman rank correlation, written out so scipy is not a dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Sharpness-gap sweep: WHEN does the raw coefficient stop being an attribution?
# ---------------------------------------------------------------------------
def sweep_sharpness_gap(emb, y, tr, va, te, budgets, n_trials=300, min_spread=1.0):
    """
    The mixture of Section 3.4 sits at one point in a two-dimensional space:
    how far apart the experts are in competence, and how far apart they are in
    posterior sharpness.  A single mixture cannot say which of the two drives
    the failure of the raw coefficients.  We therefore build a family of
    mixtures by giving each expert a different slice of its embedding -- a
    smaller slice makes a weaker and flatter expert -- and ask, across the
    family, how often each attribution reproduces the single-expert ordering
    as a function of the sharpness gap between the experts.

    All heads come from the cached embeddings, so the whole sweep costs one
    fit per (stream, budget) pair rather than one per mixture.
    """
    rng = np.random.default_rng(SEED)
    order = {s: rng.permutation(emb[s].shape[1]) for s in STREAMS}

    cache = {}
    for s in STREAMS:
        for d in budgets:
            cols = order[s][:d]
            pv, pt = fit_expert(emb[s][tr][:, cols], y[tr],
                                [emb[s][va][:, cols], emb[s][te][:, cols]])
            cache[(s, d)] = (
                pv, pt, f1_score(y[te], pt.argmax(1), average="macro") * 100
            )
        print(f"  heads fitted for stream '{s}'", flush=True)

    rows = []
    for dc in budgets:
        for dt in budgets:
            for ds in budgets:
                picks = [(STREAMS[0], dc), (STREAMS[1], dt), (STREAMS[2], ds)]
                probs_val = [cache[k][0] for k in picks]
                probs_test = [cache[k][1] for k in picks]
                solo = np.array([cache[k][2] for k in picks])
                gaps = np.diff(np.sort(solo))
                if gaps.min() < min_spread:
                    continue   # near-ties: "the ordering" is not a real target
                alpha, _ = dirichlet_search(probs_val, y[va], n_trials=n_trials)
                idx = np.arange(len(y[te]))
                eff = np.array(
                    [a * p[idx, y[te]].mean() for a, p in zip(alpha, probs_test)]
                )
                maxp = np.array([p.max(1).mean() for p in probs_test])
                truth = np.argsort(-solo)
                rows.append(dict(
                    dims_color=dc, dims_texture=dt, dims_shape=ds,
                    solo_spread=float(solo.max() - solo.min()),
                    sharpness_gap=float(maxp.max() - maxp.min()),
                    raw_recovers_order=bool((np.argsort(-alpha) == truth).all()),
                    effective_recovers_order=bool((np.argsort(-eff) == truth).all()),
                    raw_spearman=spearman(alpha, solo),
                    effective_spearman=spearman(eff, solo),
                ))
    return pd.DataFrame(rows)


def report_sweep(frame):
    banner("Sharpness-gap sweep")
    edges = [0.0, 0.05, 0.15, 0.30, 1.01]
    print(f"{'sharpness gap':>16s} {'mixtures':>9s} {'raw OK':>8s} {'effective OK':>13s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = frame[(frame.sharpness_gap >= lo) & (frame.sharpness_gap < hi)]
        if not len(sub):
            continue
        print(f"{lo:6.2f} - {hi:5.2f} {len(sub):9d} "
              f"{sub.raw_recovers_order.mean() * 100:7.0f}% "
              f"{sub.effective_recovers_order.mean() * 100:12.0f}%")
    print(f"\noverall: raw {frame.raw_recovers_order.mean() * 100:.0f}%, "
          f"effective {frame.effective_recovers_order.mean() * 100:.0f}% "
          f"of {len(frame)} mixtures")


def analyse(probs_val, probs_test, y_val, y_test, solo_f1, label, calibrated=False):
    alpha, val_f1 = dirichlet_search(probs_val, y_val)
    fused = sum(a * p for a, p in zip(alpha, probs_test))
    n = len(y_test)
    idx = np.arange(n)
    effective = np.array([a * p[idx, y_test].mean() for a, p in zip(alpha, probs_test)])
    share = effective / effective.sum() * 100
    maxp = np.array([p.max(1).mean() for p in probs_test])
    entropy = np.array(
        [(-p * np.log(np.clip(p, 1e-12, None))).sum(1).mean() for p in probs_test]
    )
    order = np.argsort(-np.asarray(solo_f1))
    row = dict(
        condition=label,
        calibrated=calibrated,
        fused_test_accuracy=accuracy_score(y_test, fused.argmax(1)) * 100,
        fused_test_macro_f1=f1_score(y_test, fused.argmax(1), average="macro") * 100,
        val_macro_f1=val_f1,
    )
    for i, s in enumerate(STREAMS):
        row[f"solo_f1_{s}"] = solo_f1[i]
        row[f"alpha_{s}"] = alpha[i]
        row[f"effective_{s}"] = effective[i]
        row[f"share_{s}"] = share[i]
        row[f"maxp_{s}"] = maxp[i]
        row[f"entropy_{s}"] = entropy[i]
    row["raw_recovers_order"] = bool((np.argsort(-alpha) == order).all())
    row["effective_recovers_order"] = bool((np.argsort(-share) == order).all())
    row["raw_spearman"] = spearman(alpha, solo_f1)
    row["effective_spearman"] = spearman(share, solo_f1)
    return row


def run_one_split(emb, y, tr, va, te, label):
    """Fit the three experts on one partition and analyse the mixture."""
    probs_val, probs_test, probs_val_raw, solo = [], [], [], []
    for s in STREAMS:
        pv, pt = fit_expert(emb[s][tr], y[tr], [emb[s][va], emb[s][te]])
        probs_val.append(pv)
        probs_test.append(pt)
        probs_val_raw.append(pv)
        solo.append(f1_score(y[te], pt.argmax(1), average="macro") * 100)

    plain = analyse(probs_val, probs_test, y[va], y[te], solo, label, calibrated=False)

    temps = [temperature_fit(p, y[va]) for p in probs_val_raw]
    cal_val = [apply_temperature(p, t) for p, t in zip(probs_val, temps)]
    cal_test = [apply_temperature(p, t) for p, t in zip(probs_test, temps)]
    calibrated = analyse(cal_val, cal_test, y[va], y[te], solo, label, calibrated=True)
    for i, s in enumerate(STREAMS):
        calibrated[f"temperature_{s}"] = temps[i]
        calibrated[f"ece_before_{s}"] = ece(probs_test[i], y[te])
        calibrated[f"ece_after_{s}"] = ece(cal_test[i], y[te])
    return plain, calibrated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", type=int, default=9)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--min-spread", type=float, default=1.0,
                    help="minimum Macro-F1 gap between consecutive experts for a\n                          sweep mixture to be counted")
    ap.add_argument("--budgets", default="16,64,256,1280",
                    help="embedding-dimension budgets used to build the sweep")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    env = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.processor(),
        torch=torch.__version__,
        torchvision=__import__("torchvision").__version__,
        numpy=np.__version__,
        sklearn=__import__("sklearn").__version__,
        opencv=cv2.__version__,
        device=device,
    )
    (RESULT_DIR / "deep_moe_environment.json").write_text(json.dumps(env, indent=2))

    banner(f"Deep mixture-of-experts attribution control  (device: {device})")
    for k, v in env.items():
        print(f"  {k:12s}: {v}")

    pool = load_pool(args.data_root, args.limit)
    print(f"\nimages: {len(pool)}   classes: {pool['label_id'].nunique()}")

    emb = compute_embeddings(
        pool, device, args.grid, args.batch_size, args.workers, args.recompute
    )
    y = pool["label_id"].to_numpy()

    rows = []
    # 1. the paper's own partition
    tr = np.where(pool["split"] == "train")[0]
    va = np.where(pool["split"] == "val")[0]
    te = np.where(pool["split"] == "test")[0]
    banner("Paper partition")
    plain, calibrated = run_one_split(emb, y, tr, va, te, "paper-split")
    rows += [plain, calibrated]
    report(plain, calibrated)

    # 2. independent stratified re-splits, same 70/15/15 recipe
    for seed in range(args.splits):
        idx = np.arange(len(y))
        tr, rest = train_test_split(
            idx, train_size=0.70, stratify=y, random_state=seed
        )
        va, te = train_test_split(
            rest, train_size=0.50, stratify=y[rest], random_state=seed
        )
        p, c = run_one_split(emb, y, tr, va, te, f"resplit-{seed}")
        rows += [p, c]
        print(
            f"  re-split {seed}: raw {'OK ' if p['raw_recovers_order'] else 'NO '}"
            f" effective {'OK ' if p['effective_recovers_order'] else 'NO '}"
            f" fused Macro-F1 {p['fused_test_macro_f1']:.2f}%"
        )

    # 3. the sharpness-gap sweep, on the paper partition
    if not args.no_sweep:
        banner("Sharpness-gap sweep (one head per stream and budget)")
        tr = np.where(pool["split"] == "train")[0]
        va = np.where(pool["split"] == "val")[0]
        te = np.where(pool["split"] == "test")[0]
        budgets = [int(b) for b in args.budgets.split(",")]
        sweep = sweep_sharpness_gap(emb, y, tr, va, te, budgets,
                                    min_spread=args.min_spread)
        sweep.to_csv(RESULT_DIR / "deep_moe_sweep.csv", index=False)
        report_sweep(sweep)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_DIR / "deep_moe_per_split.csv", index=False)
    frame[frame.calibrated].to_csv(RESULT_DIR / "deep_moe_calibration.csv", index=False)

    plain_rows = frame[~frame.calibrated]
    cal_rows = frame[frame.calibrated]
    summary = dict(
        n_splits=len(plain_rows),
        raw_recovers=int(plain_rows.raw_recovers_order.sum()),
        effective_recovers=int(plain_rows.effective_recovers_order.sum()),
        raw_spearman_mean=plain_rows.raw_spearman.mean(),
        raw_spearman_sd=plain_rows.raw_spearman.std(),
        effective_spearman_mean=plain_rows.effective_spearman.mean(),
        effective_spearman_sd=plain_rows.effective_spearman.std(),
        raw_recovers_calibrated=int(cal_rows.raw_recovers_order.sum()),
        effective_recovers_calibrated=int(cal_rows.effective_recovers_order.sum()),
    )
    for s in STREAMS:
        summary[f"solo_f1_{s}"] = plain_rows[f"solo_f1_{s}"].mean()
        summary[f"alpha_{s}"] = plain_rows[f"alpha_{s}"].mean()
        summary[f"share_{s}"] = plain_rows[f"share_{s}"].mean()
        summary[f"maxp_{s}"] = plain_rows[f"maxp_{s}"].mean()
        summary[f"temperature_{s}"] = cal_rows[f"temperature_{s}"].mean()
    pd.DataFrame([summary]).to_csv(RESULT_DIR / "deep_moe_summary.csv", index=False)

    banner("Summary over all partitions")
    print(
        f"raw coefficients reproduce the single-stream ordering in "
        f"{summary['raw_recovers']}/{summary['n_splits']} partitions "
        f"(Spearman {summary['raw_spearman_mean']:.2f} ± {summary['raw_spearman_sd']:.2f})"
    )
    print(
        f"effective contribution reproduces it in "
        f"{summary['effective_recovers']}/{summary['n_splits']} "
        f"(Spearman {summary['effective_spearman_mean']:.2f} ± "
        f"{summary['effective_spearman_sd']:.2f})"
    )
    print(
        f"after temperature scaling: raw {summary['raw_recovers_calibrated']}"
        f"/{summary['n_splits']}, effective "
        f"{summary['effective_recovers_calibrated']}/{summary['n_splits']}"
    )
    print("\nwritten to results/deep_moe_summary.csv, deep_moe_per_split.csv, "
          "deep_moe_calibration.csv, deep_moe_environment.json")


def report(plain, calibrated):
    print(f"\n{'stream':10s} {'solo F1':>9s} {'alpha':>8s} {'share %':>9s} "
          f"{'max P':>8s} {'entropy':>9s} {'T':>7s}")
    for s in STREAMS:
        print(
            f"{s:10s} {plain['solo_f1_' + s]:9.2f} {plain['alpha_' + s]:8.3f} "
            f"{plain['share_' + s]:9.1f} {plain['maxp_' + s]:8.3f} "
            f"{plain['entropy_' + s]:9.3f} {calibrated['temperature_' + s]:7.3f}"
        )
    print(f"\nfused test Macro-F1 {plain['fused_test_macro_f1']:.2f}%")
    print(f"raw coefficients recover the ordering : {plain['raw_recovers_order']}")
    print(f"effective contribution recovers it    : {plain['effective_recovers_order']}")


if __name__ == "__main__":
    main()

"""
End-to-end latency of the handcrafted pipeline.

Section 3.7 currently reports 0.94 ms per image for the SVM *decision function*
and states plainly that feature extraction is excluded.  That was defensible
while the comparison with deep models was indirect.  It is not defensible now
that a MobileNetV2 control has been measured at 11.6 ms per image on the same
CPU: the only honest comparison is image in, label out, for both systems.

This script measures the missing half.  It re-implements the extractors exactly
as they appear in the original pipeline notebook -- same IMG_SIZE, same HSV bin
count, same GLCM distances, angles and properties, same LBP parameters, same
HSV mask thresholds and morphology -- so the timings correspond to the features
actually behind Tables 4 to 9, not to a re-optimised version.

Usage
-----
    python src/run_timing.py --data-root /path/to/PlantVillage

Add --check to verify that the re-implementation reproduces the stored feature
CSVs before trusting the timings.

Outputs
-------
    results/timing_handcrafted.csv
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.stats as st
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from common import RESULT_DIR, SPLIT_DIR, banner, feature_matrix, load_all, make_svm


def rebase(stored_path, data_root):
    """Map a path recorded during the original run onto the local dataset copy.

    Duplicated from run_deep_baseline.py on purpose, so this script runs without
    torch installed.
    """
    parts = Path(stored_path).parts
    anchor = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "PlantVillage":
            anchor = i
            break
    tail = parts[anchor + 1:] if anchor is not None else parts[-2:]
    return Path(data_root).joinpath(*tail)


IMG_SIZE = (256, 256)
HSV_BINS = 16
EPS = 1e-8

DISTANCES = [1, 2]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS = LBP_POINTS + 2


# --------------------------------------------------------------------------
# Extractors — identical to the original notebook
# --------------------------------------------------------------------------
def color_block(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    moments = []
    for c in range(3):
        ch = rgb[..., c].reshape(-1)
        moments.extend([np.mean(ch), np.std(ch), st.skew(ch)])

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hists = []
    for i, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        hist = cv2.calcHist([hsv], [i], None, [HSV_BINS], [lo, hi]).flatten()
        hists.append(hist / (hist.sum() + EPS))
    return np.concatenate([np.array(moments, dtype=np.float32), np.concatenate(hists)])


def texture_block(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    levels = 16
    gray_q = (gray / (256 / levels)).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=DISTANCES, angles=ANGLES,
                        levels=levels, symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        feats.extend([vals.mean(), vals.std()])

    lbp = local_binary_pattern(gray.astype(np.float32) / 255.0,
                               LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_BINS + 1))
    hist = hist.astype(np.float32)
    return np.concatenate([np.array(feats, dtype=np.float32), hist / (hist.sum() + EPS)])


def shape_block(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([95, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(8, dtype=np.float32)

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    hull_area = float(cv2.contourArea(cv2.convexHull(c)))
    return np.array([
        area,
        perimeter,
        (4.0 * np.pi * area) / (perimeter * perimeter + EPS),
        float(w) / (float(h) + EPS),
        area / (float(w * h) + EPS),
        area / (hull_area + EPS),
        np.sqrt(4.0 * area / (np.pi + EPS)),
        perimeter / (2.0 * np.sqrt(np.pi * area) + EPS),
    ], dtype=np.float32)


def full_vector(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"could not read {path}")
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return np.concatenate([color_block(img), texture_block(img), shape_block(img)])


# --------------------------------------------------------------------------
def timed(fn, images, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        for img in images:
            fn(img)
    return (time.perf_counter() - start) / (repeats * len(images)) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--n", type=int, default=200,
                        help="how many test images to time over")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--check", action="store_true",
                        help="verify the re-implementation against the stored feature CSVs")
    args = parser.parse_args()

    banner("End-to-end latency of the handcrafted pipeline")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv").sort_values("path").reset_index(drop=True)
    sample = test_df.sample(n=min(args.n, len(test_df)), random_state=0)
    files = [rebase(p, args.data_root) for p in sample["path"]]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(files)} sampled test images not found under "
            f"{args.data_root}.\nFirst missing: {missing[0]}"
        )
    print(f"timing over {len(files)} test images, {args.repeats} repeats\n")

    # decode + resize, measured separately because both systems pay it
    start = time.perf_counter()
    for _ in range(args.repeats):
        decoded = [cv2.resize(cv2.imread(str(f)), IMG_SIZE, interpolation=cv2.INTER_AREA)
                   for f in files]
    decode_ms = (time.perf_counter() - start) / (args.repeats * len(files)) * 1000

    color_ms = timed(color_block, decoded, args.repeats)
    texture_ms = timed(texture_block, decoded, args.repeats)
    shape_ms = timed(shape_block, decoded, args.repeats)
    features_ms = color_ms + texture_ms + shape_ms

    # classifier, on the same machine, for a like-for-like total
    train, val, test, columns = load_all()
    svm = make_svm()
    svm.fit(feature_matrix(train, columns), train["label_id"].values)
    X = feature_matrix(test, columns)[: len(files)]
    start = time.perf_counter()
    for _ in range(args.repeats):
        for row in X:
            svm.predict(row.reshape(1, -1))
    clf_single_ms = (time.perf_counter() - start) / (args.repeats * len(X)) * 1000
    start = time.perf_counter()
    for _ in range(args.repeats):
        svm.predict(X)
    clf_batched_ms = (time.perf_counter() - start) / (args.repeats * len(X)) * 1000

    total_single = decode_ms + features_ms + clf_single_ms
    total_batched = decode_ms + features_ms + clf_batched_ms

    print(f"decode + resize      : {decode_ms:8.2f} ms/image")
    print(f"color   (57 dims)    : {color_ms:8.2f} ms/image")
    print(f"texture (22 dims)    : {texture_ms:8.2f} ms/image")
    print(f"shape   (8 dims)     : {shape_ms:8.2f} ms/image")
    print(f"  feature total      : {features_ms:8.2f} ms/image")
    print(f"SVM decision, single : {clf_single_ms:8.2f} ms/image")
    print(f"SVM decision, batched: {clf_batched_ms:8.2f} ms/image")
    print(f"\nEND-TO-END, single   : {total_single:8.2f} ms/image")
    print(f"END-TO-END, batched  : {total_batched:8.2f} ms/image")
    print("\nCompare against the MobileNetV2 control in "
          "results/table7_deep_baseline.csv (cpu_ms_single / cpu_ms_batched).")

    pd.DataFrame([dict(
        n_images=len(files),
        decode_ms=round(decode_ms, 3),
        color_ms=round(color_ms, 3),
        texture_ms=round(texture_ms, 3),
        shape_ms=round(shape_ms, 3),
        features_ms=round(features_ms, 3),
        svm_single_ms=round(clf_single_ms, 3),
        svm_batched_ms=round(clf_batched_ms, 3),
        end_to_end_single_ms=round(total_single, 3),
        end_to_end_batched_ms=round(total_batched, 3),
    )]).to_csv(RESULT_DIR / "timing_handcrafted.csv", index=False)
    print("\nwrote results/timing_handcrafted.csv")

    if args.check:
        banner("Reproduction check against the stored feature CSVs")
        flat = [c for g in ("color", "texture", "shape") for c in columns[g]]
        stored = test.set_index("path")
        ok = 0
        checked = 0
        for f, p in zip(files[:20], sample["path"][:20]):
            if p not in stored.index:
                continue
            checked += 1
            mine = full_vector(f)
            theirs = stored.loc[p, flat].to_numpy(dtype=float)
            if np.allclose(mine, theirs, rtol=1e-3, atol=1e-4):
                ok += 1
            else:
                worst = int(np.argmax(np.abs(mine - theirs)))
                print(f"  mismatch on {Path(p).name}: worst column "
                      f"{flat[worst]} {mine[worst]:.5f} vs {theirs[worst]:.5f}")
        print(f"  {ok}/{checked} sampled images reproduce the stored features")


if __name__ == "__main__":
    main()

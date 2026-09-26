"""
Background control for the JAIDM leaf-disease paper (reviewer point: is the colour
dominance caused by the PlantVillage background?).

For every image listed in data/splits/{train,val,test}.csv this script recomputes the
57 colour features of Section 2.2 twice:
  * LEAF  -- only over pixels inside the leaf (background removed);
  * BG    -- only over background pixels (the leaf removed).
If colour dominance were a background artefact, the BG features would classify well
and the LEAF features would lose most of the colour expert's accuracy.

Leaf mask: Otsu threshold on HSV saturation (PlantVillage backgrounds are grey and
unsaturated), largest connected component, holes filled -- so brown/yellow lesions
inside the leaf outline stay in the mask (a green-only mask would drop them).

Run from the repository root (needs the Kaggle images at the paths in the split CSVs;
edit IMAGE_ROOT below if they moved):
    pip install opencv-python scipy pandas numpy
    python extract_background_control.py
Output: data/features/color_leaf_features_{split}.csv and color_bg_features_{split}.csv
(~2 MB each). Send these six CSVs back; the analysis runs on them directly.
Runtime: roughly 5-10 minutes on a laptop.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "data" / "features"
IMAGE_ROOT = None  # e.g. Path("/Users/you/.cache/kagglehub/datasets/emmarex/plantdisease/versions/1/PlantVillage")
IMG_SIZE = (256, 256)
HSV_BINS = 16
EPS = 1e-8


def leaf_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return filled > 0


def colour_features(img_bgr, sel):
    if sel.sum() < 50:  # degenerate mask: fall back to all pixels, flagged below
        sel = np.ones(sel.shape, bool)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[sel].astype(np.float64) / 255.0
    moments = []
    for c in range(3):
        ch = rgb[:, c]
        moments += [ch.mean(), ch.std(), st.skew(ch) if ch.std() > 0 else 0.0]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[sel]
    hists = []
    for i, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        h, _ = np.histogram(hsv[:, i], bins=HSV_BINS, range=(lo, hi))
        hists.append(h / (h.sum() + EPS))
    return np.concatenate([moments, *hists])


def resolve(p):
    if IMAGE_ROOT is None:
        return p
    parts = Path(p).parts
    return str(IMAGE_ROOT / parts[-2] / parts[-1])


def main():
    names = [f"cm_{c}_{s}" for c in "RGB" for s in ("mean", "std", "skew")] + \
            [f"hsv_{c}_bin{i}" for c in "HSV" for i in range(HSV_BINS)]
    for split in ("train", "val", "test"):
        df = pd.read_csv(SPLITS / f"{split}.csv")
        leaf, bg, frac = [], [], []
        for p in df["path"]:
            img = cv2.imread(resolve(p))
            if img is None:
                raise RuntimeError(f"cannot read {p}")
            img = cv2.resize(img, IMG_SIZE)
            m = leaf_mask(img)
            frac.append(m.mean())
            leaf.append(colour_features(img, m))
            bg.append(colour_features(img, ~m))
        for tag, X in (("leaf", leaf), ("bg", bg)):
            out = df[["path", "label", "label_id"]].copy()
            out[names] = np.vstack(X)
            out["leaf_fraction"] = frac
            out.to_csv(OUT / f"color_{tag}_features_{split}.csv", index=False)
        f = np.array(frac)
        print(f"{split}: {len(df)} images, leaf fraction {f.mean():.2f} +/- {f.std():.2f}, "
              f"masks <5% or >95% of image: {int(((f < .05) | (f > .95)).sum())}")


if __name__ == "__main__":
    main()

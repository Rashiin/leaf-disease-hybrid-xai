"""
MobileNetV2 transfer-learning baseline on the *same* 15-class split
-> the missing "deep model" row of Table 7, plus a paired significance
   test against the handcrafted RBF-SVM of Table 4.

Why this script exists
----------------------
Every number in the paper compares the proposed handcrafted pipeline
against deep models *reported in other papers*, on other class counts and
other evaluation protocols.  A referee can reasonably object that no deep
model was ever run on this paper's own subset and split.  This script
closes that gap: it trains a standard ImageNet-pretrained MobileNetV2 on
exactly the partition stored in data/splits/, selects on validation
Macro-F1 (the same protocol as Section 2.6), evaluates once on the
held-out test set, and reports the same quantities Section 3.7 reports for
the SVM -- parameter count, model size, CPU latency -- so the efficiency
claim becomes a measured comparison rather than a citation.

It deliberately uses **no data augmentation** by default, because the
handcrafted pipeline uses none either; enabling it would make the
comparison unfair in the deep model's favour.  Pass --augment if you want
the augmented number as well.

Usage
-----
    python src/run_deep_baseline.py --data-root /path/to/PlantVillage

`--data-root` is the directory that contains the 15 class sub-directories.
The paths stored in data/splits/*.csv are absolute paths from the machine
the study was run on; they are rebased onto --data-root automatically by
keeping everything after the last "PlantVillage/" component.

Requires (not in requirements.txt -- this script is supplementary):
    pip install torch torchvision pillow

Outputs
-------
    results/table7_deep_baseline.csv          headline numbers + cost
    results/deep_baseline_test_predictions.csv  per-image predictions
    results/deep_vs_svm_mcnemar.csv           paired test against the SVM
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from common import RESULT_DIR, SPLIT_DIR, banner, feature_matrix, load_all, make_svm

SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Split loading and path rebasing
# --------------------------------------------------------------------------
def rebase(stored_path, data_root):
    """
    Map an absolute path recorded during the original run onto the local
    dataset copy.  Everything after the last 'PlantVillage' component is
    kept, so <class_dir>/<file>.JPG survives intact.
    """
    parts = Path(stored_path).parts
    anchor = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "PlantVillage":
            anchor = i
            break
    tail = parts[anchor + 1:] if anchor is not None else parts[-2:]
    return Path(data_root).joinpath(*tail)


def load_splits(data_root, limit=None):
    frames = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{split}.csv").sort_values("path")
        frame = frame.reset_index(drop=True)
        if limit:
            frame = (
                frame.groupby("label_id", group_keys=False)
                .head(max(1, limit // frame["label_id"].nunique()))
                .reset_index(drop=True)
            )
        frame["file"] = [rebase(p, data_root) for p in frame["path"]]
        missing = [f for f in frame["file"] if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(frame)} '{split}' images not found under "
                f"{data_root}.\nFirst missing: {missing[0]}\n"
                "Check that --data-root points at the directory holding the 15 "
                "class sub-directories."
            )
        frames[split] = frame
    return frames["train"], frames["val"], frames["test"]


# --------------------------------------------------------------------------
# Dataset
#
# LeafDataset is defined at module level on purpose.  macOS starts DataLoader
# workers with "spawn", which pickles the dataset object; a class defined
# inside a function cannot be pickled and the run dies as soon as
# --workers is greater than zero.
# --------------------------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, files, labels, pipeline):
        self.files = list(files)
        self.labels = np.asarray(labels)
        self.pipeline = pipeline

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        return self.pipeline(image), int(self.labels[index])


def build_pipeline(train_mode, augment):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train_mode and augment:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def build_dataset(frame, train_mode, augment):
    return LeafDataset(
        frame["file"], frame["label_id"].to_numpy(), build_pipeline(train_mode, augment)
    )


def build_model(n_classes, pretrained):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, loader, device):
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            outputs.append(logits.argmax(1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def measure_latency(model, repeats=50):
    """Single-image and batched CPU latency, matching Section 3.7's protocol."""
    model = model.to("cpu").eval()
    single = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    batch = torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        for _ in range(3):
            model(single)
        start = time.perf_counter()
        for _ in range(repeats):
            model(single)
        one_at_a_time = (time.perf_counter() - start) / repeats * 1000
        model(batch)
        start = time.perf_counter()
        for _ in range(3):
            model(batch)
        batched = (time.perf_counter() - start) / (3 * 32) * 1000
    return one_at_a_time, batched


# --------------------------------------------------------------------------
# Paired comparison against the handcrafted SVM
# --------------------------------------------------------------------------
def mcnemar(correct_a, correct_b):
    """
    Exact McNemar test on the two discordant cells.  `correct_a` and
    `correct_b` are boolean arrays over the same test images.
    """
    from scipy.stats import binomtest

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def svm_test_predictions():
    """Re-run the Table 4 baseline so the comparison is paired image by image."""
    train, val, test, columns = load_all()
    model = make_svm()
    model.fit(feature_matrix(train, columns), train["label_id"].values)
    pred = model.predict(feature_matrix(test, columns))
    return test["path"].values, test["label_id"].values, pred


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="directory holding the 15 PlantVillage class folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--augment", action="store_true",
                        help="enable flip/rotation augmentation (off by default so "
                             "the comparison with the handcrafted pipeline is fair)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="train from scratch instead of ImageNet initialisation")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only ~N images per split (smoke test)")
    parser.add_argument("--skip-svm", action="store_true",
                        help="skip the paired McNemar comparison")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = pick_device(args.device)
    banner(f"MobileNetV2 baseline on the Table 4 split  (device: {device})")

    train_df, val_df, test_df = load_splits(args.data_root, args.limit or None)
    n_classes = int(pd.concat([train_df, val_df, test_df])["label_id"].nunique())
    print(f"train / val / test : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"classes            : {n_classes}")
    print(f"augmentation       : {'on' if args.augment else 'off'}")
    print(f"initialisation     : {'random' if args.no_pretrained else 'ImageNet'}")

    loaders = {}
    for name, frame, shuffle in (
        ("train", train_df, True),
        ("val", val_df, False),
        ("test", test_df, False),
    ):
        loaders[name] = DataLoader(
            build_dataset(frame, name == "train", args.augment),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
        )

    model = build_model(n_classes, not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    counts = train_df["label_id"].value_counts().sort_index().to_numpy()
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state, best_val_f1, history = None, -1.0, []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            running += loss.item() * images.size(0)

        val_pred, val_true = predict(model, loaders["val"], device)
        val_f1 = f1_score(val_true, val_pred, average="macro") * 100
        val_acc = accuracy_score(val_true, val_pred) * 100
        history.append(dict(epoch=epoch, train_loss=running / len(train_df),
                            val_accuracy=val_acc, val_macro_f1=val_f1))
        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1, marker = val_f1, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  loss {running / len(train_df):.4f}  "
              f"val acc {val_acc:.2f}%  val Macro-F1 {val_f1:.2f}%{marker}")
    train_seconds = time.perf_counter() - train_start

    model.load_state_dict(best_state)
    val_pred, val_true = predict(model, loaders["val"], device)
    test_pred, test_true = predict(model, loaders["test"], device)

    test_acc = accuracy_score(test_true, test_pred) * 100
    test_f1 = f1_score(test_true, test_pred, average="macro") * 100
    val_acc = accuracy_score(val_true, val_pred) * 100

    banner("Result — the deep row of Table 7")
    print(f"validation : accuracy {val_acc:.2f}%   Macro-F1 {best_val_f1:.2f}%")
    print(f"test       : accuracy {test_acc:.2f}%   Macro-F1 {test_f1:.2f}%")

    size_mb = n_params * 4 / 1024 ** 2
    single_ms, batched_ms = measure_latency(model)
    print(f"\nparameters      : {n_params:,}  ({size_mb:.1f} MB at float32)")
    print(f"training time   : {train_seconds / 60:.1f} min on {device}")
    print(f"CPU inference   : {single_ms:.2f} ms/image single, "
          f"{batched_ms:.2f} ms/image batched")

    pd.DataFrame(history).to_csv(RESULT_DIR / "deep_baseline_history.csv", index=False)
    pd.DataFrame([dict(
        model="MobileNetV2 (ImageNet init)" if not args.no_pretrained else "MobileNetV2 (scratch)",
        augmentation="on" if args.augment else "off",
        epochs=args.epochs,
        val_accuracy=round(val_acc, 2),
        val_macro_f1=round(best_val_f1, 2),
        test_accuracy=round(test_acc, 2),
        test_macro_f1=round(test_f1, 2),
        parameters=n_params,
        size_mb=round(size_mb, 1),
        train_minutes=round(train_seconds / 60, 2),
        cpu_ms_single=round(single_ms, 2),
        cpu_ms_batched=round(batched_ms, 2),
    )]).to_csv(RESULT_DIR / "table7_deep_baseline.csv", index=False)

    predictions = test_df[["path", "label", "label_id"]].copy()
    predictions["deep_pred"] = test_pred
    predictions.to_csv(RESULT_DIR / "deep_baseline_test_predictions.csv", index=False)

    if args.skip_svm or args.limit:
        print("\n(paired SVM comparison skipped)")
        return

    banner("Paired comparison against the handcrafted SVM (same test images)")
    svm_paths, svm_true, svm_pred = svm_test_predictions()
    order = pd.Series(range(len(svm_paths)), index=svm_paths)
    index = order.reindex(predictions["path"]).to_numpy()
    if np.isnan(index).any():
        raise ValueError("test image paths do not line up between the two models")
    index = index.astype(int)
    svm_correct = (svm_pred[index] == svm_true[index])
    deep_correct = (test_pred == test_true)

    svm_only, deep_only, p = mcnemar(svm_correct, deep_correct)
    svm_acc = svm_correct.mean() * 100
    deep_acc = deep_correct.mean() * 100
    print(f"SVM  correct : {svm_correct.sum():4d} / {len(svm_correct)}  ({svm_acc:.2f}%)")
    print(f"deep correct : {deep_correct.sum():4d} / {len(deep_correct)}  ({deep_acc:.2f}%)")
    print(f"SVM right & deep wrong : {svm_only}")
    print(f"deep right & SVM wrong : {deep_only}")
    print(f"exact McNemar p        : {p:.4g}")
    print("\nInterpretation: p >= 0.05 means the two models are not "
          "distinguishable on this test set, which is the strongest form the "
          "'competitive with deep models' claim can honestly take.")

    pd.DataFrame([dict(
        svm_accuracy=round(svm_acc, 2),
        deep_accuracy=round(deep_acc, 2),
        svm_right_deep_wrong=svm_only,
        deep_right_svm_wrong=deep_only,
        mcnemar_p=p,
    )]).to_csv(RESULT_DIR / "deep_vs_svm_mcnemar.csv", index=False)


if __name__ == "__main__":
    main()

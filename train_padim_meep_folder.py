#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
anomalib v2.x compatible PaDiM training on folder dataset.

Expected dataset layout:
  data/
    train/good/*.png
    test/good/*.png
    test/anomaly/*.png

Run:
  conda activate meep-anomalib
  python train_padim_meep_folder_v2.py --root data --name meep_plane_scatter --epochs 1

Outputs:
  results/<name>/... (images/anomaly maps etc.)
"""

import argparse
from pathlib import Path

import torch

from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data.utils import ValSplitMode, TestSplitMode

# torchvision transforms v2 (anomalib v2 uses torchvision v2 API)
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype


def build_aug(image_size: int):
    """Minimal preprocessing: uint8 -> float32[0,1] and resize."""
    return Compose([
        ToImage(),
        Resize((image_size, image_size)),
        ToDtype(torch.float32, scale=True),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data", help="dataset root folder (contains train/ and test/)")
    ap.add_argument("--name", type=str, default="meep_padim", help="experiment name for results")
    ap.add_argument("--results", type=str, default="results", help="results output directory")
    ap.add_argument("--image_size", type=int, default=256, help="resize images to this square size")
    ap.add_argument("--epochs", type=int, default=1, help="epochs (PaDiM typically works with few epochs)")
    ap.add_argument("--batch", type=int, default=16, help="train batch size")
    ap.add_argument("--eval_batch", type=int, default=16, help="eval batch size")
    ap.add_argument("--workers", type=int, default=8, help="num dataloader workers")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    assert root.exists(), f"Dataset root not found: {root}"

    aug = build_aug(args.image_size)

    # Folder datamodule (v2): use augmentations instead of transform
    datamodule = Folder(
        name=args.name,
        root=str(root),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/anomaly",
        extensions=(".png", ".jpg", ".jpeg"),
        train_batch_size=args.batch,
        eval_batch_size=args.eval_batch,
        num_workers=args.workers,
        # validation split is built from normal images (safe)
        val_split_mode=ValSplitMode.SYNTHETIC,
        val_split_ratio=0.2,
        # test comes from the provided folders
        test_split_mode=TestSplitMode.FROM_DIR,
        augmentations=aug,              # <- ここがポイント
        seed=args.seed,
    )

    model = Padim(backbone="resnet18")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    engine = Engine(
        default_root_dir=args.results,
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
    )

    print(f"[Info] root        : {root}")
    print(f"[Info] results     : {Path(args.results).resolve()}")
    print(f"[Info] accelerator : {accelerator}")
    print(f"[Info] image_size  : {args.image_size}")
    print(f"[Info] epochs      : {args.epochs}")

    engine.fit(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)

    # predict: data_path can point to a folder; easiest is the test folder
    engine.predict(model=model, data_path=str(root / "test"))

    print("\n[Done] Check results directory for anomaly maps / predictions.")
    print(f"       -> {Path(args.results).resolve()}")


if __name__ == "__main__":
    main()

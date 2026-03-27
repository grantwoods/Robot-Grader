#!/usr/bin/env python3
"""Train a YOLOv8 classification model for fruit grading.

Usage:
    python scripts/train_model.py --fruit-type apple
    python scripts/train_model.py --fruit-type apple --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path


GRADES = ["trash", "choice", "fancy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 classifier for fruit grading.",
    )
    parser.add_argument("--fruit-type", required=True, help="Type of fruit to train on.")
    parser.add_argument("--data-dir", default="data", help="Root data directory (default: data).")
    parser.add_argument("--output-dir", default="models", help="Where to save the trained model (default: models).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def gather_images(data_root: Path) -> dict[str, list[Path]]:
    """Collect all images per grade class, mixing camera angles together."""
    images_by_grade: dict[str, list[Path]] = {}
    for grade in GRADES:
        grade_dir = data_root / grade
        if not grade_dir.exists():
            print(f"Warning: grade directory not found: {grade_dir}")
            images_by_grade[grade] = []
            continue
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            imgs.extend(grade_dir.rglob(ext))
        images_by_grade[grade] = sorted(imgs)
        print(f"  {grade}: {len(imgs)} images")
    return images_by_grade


def build_split_dirs(
    images_by_grade: dict[str, list[Path]],
    tmpdir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """Create train/val directory structure expected by YOLOv8 classification."""
    random.seed(seed)
    for grade, imgs in images_by_grade.items():
        if not imgs:
            continue
        shuffled = imgs.copy()
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - val_fraction)))
        train_imgs = shuffled[:split_idx]
        val_imgs = shuffled[split_idx:]

        train_dir = tmpdir / "train" / grade
        val_dir = tmpdir / "val" / grade
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / img.name)

        print(f"  {grade}: {len(train_imgs)} train, {len(val_imgs)} val")

    return tmpdir


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_dir) / args.fruit_type
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Training fruit classifier: {args.fruit_type} ===")
    print(f"Data dir   : {data_root}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Image size : {args.img_size}")
    print()

    # Gather images from all camera angles.
    print("Gathering images...")
    images_by_grade = gather_images(data_root)

    total = sum(len(v) for v in images_by_grade.values())
    if total == 0:
        print(f"Error: no images found in {data_root}. Run collect_training_data.py first.")
        return

    print(f"\nTotal images: {total}")
    print()

    # Build temporary train/val split.
    with tempfile.TemporaryDirectory(prefix="yolo_cls_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        print("Building train/val split...")
        build_split_dirs(images_by_grade, tmpdir_path, seed=args.seed)
        print()

        # Train YOLOv8 classifier.
        print("Starting YOLOv8 classification training...")
        from ultralytics import YOLO

        model = YOLO("yolov8n-cls.pt")
        results = model.train(
            data=str(tmpdir_path),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
        )

        # Print results.
        print("\n=== Training Complete ===")
        if hasattr(results, "results_dict"):
            for key, value in results.results_dict.items():
                print(f"  {key}: {value}")
        elif hasattr(results, "top1") and hasattr(results, "top5"):
            print(f"  Top-1 accuracy: {results.top1:.4f}")
            print(f"  Top-5 accuracy: {results.top5:.4f}")

        # Export and save the model.
        model_save_path = output_dir / f"{args.fruit_type}_classifier.pt"
        best_weight = Path(model.trainer.best) if hasattr(model, "trainer") else None

        if best_weight and best_weight.exists():
            shutil.copy2(best_weight, model_save_path)
        else:
            # Fallback: export the model.
            model.export(format="torchscript")
            # Save the last checkpoint.
            last_weight = Path(model.trainer.last) if hasattr(model, "trainer") else None
            if last_weight and last_weight.exists():
                shutil.copy2(last_weight, model_save_path)

        print(f"\nModel saved to: {model_save_path}")


if __name__ == "__main__":
    main()

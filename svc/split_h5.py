"""Create reproducible train/val/test WSI splits from labeled Trident H5 files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import h5py
import numpy as np


def make_splits(job_dir: Path, output: Path, *, seed: int = 42,
                features_dir: Path | None = None) -> list[dict[str, object]]:
    job_dir = job_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if not job_dir.is_dir():
        raise ValueError(f"Trident job directory not found: {job_dir}")
    if features_dir is not None:
        features_dir = features_dir.expanduser().resolve()
        if not features_dir.is_dir():
            raise ValueError(f"Feature directory not found: {features_dir}")
        paths = sorted(features_dir.glob("*.h5"))
    else:
        paths = sorted(job_dir.rglob("features_*/*.h5"))
    if not paths:
        raise ValueError("No Trident feature H5 files found")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    feature_dimension: int | None = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "labels" not in handle:
                continue
            if "features" not in handle or "coords" not in handle:
                raise ValueError(f"Missing features or coords in {path}")
            features = handle["features"]
            labels = np.asarray(handle["labels"][:])
            if features.ndim != 2 or handle["coords"].shape != (len(features), 2):
                raise ValueError(f"Invalid features or coords shape in {path}")
            if labels.shape != (len(features),) or not np.isin(labels, [0, 1, 2]).all():
                raise ValueError(f"Invalid tile labels in {path}")
            cortex = int(np.count_nonzero(labels == 1))
            medullary = int(np.count_nonzero(labels == 2))
            if cortex + medullary == 0:
                continue
            if path.stem in seen:
                raise ValueError(f"Multiple labeled feature files for {path.stem}; pass --features-dir")
            seen.add(path.stem)
            if feature_dimension is None:
                feature_dimension = features.shape[1]
            elif features.shape[1] != feature_dimension:
                raise ValueError(f"Feature dimension differs in {path}; use one encoder")
            rows.append({"sample_name": path.stem, "feature_path": str(path),
                         "cortex_tiles": cortex, "medullary_tiles": medullary})

    if len(rows) < 3:
        raise ValueError("At least three labeled WSIs are needed for train, val, and test")
    random.Random(seed).shuffle(rows)
    train_count = min(len(rows) - 2, max(1, round(len(rows) * 0.70)))
    val_count = min(len(rows) - train_count - 1, max(1, round(len(rows) * 0.15)))
    for index, row in enumerate(rows):
        row["split"] = "train" if index < train_count else (
            "val" if index < train_count + val_count else "test"
        )
    rows.sort(key=lambda row: str(row["sample_name"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_name", "split", "feature_path",
                                                    "cortex_tiles", "medullary_tiles"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = make_splits(args.job_dir, args.output, seed=args.seed,
                       features_dir=args.features_dir)
    counts = {split: sum(row["split"] == split for row in rows)
              for split in ("train", "val", "test")}
    print(f"Saved {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

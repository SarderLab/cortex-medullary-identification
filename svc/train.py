"""Train, evaluate, and save the GPU PCA + RBF SVC pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BEST_C,
    BEST_N_COMPONENTS,
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MODEL_DIR,
    DEFAULT_SPLIT_XLSX,
    DEFAULT_THRESHOLD,
    LABEL_NAMES,
    RAW_LABEL_MAPPING,
)
from .data import load_labeled_dataset, load_labeled_h5_dataset
from .evaluation import (
    classification_metrics,
    labeled_predictions_dataframe,
    save_confusion_matrix,
)
from .gpu_model import (
    base_metadata,
    predict_gpu,
    save_bundle,
    train_gpu_pipeline,
    write_json,
)
from .manifest import load_h5_split_samples, load_split_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and save a GPU cortex/medulla PCA + RBF SVC.",
    )
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_SPLIT_XLSX)
    parser.add_argument("--manifest", type=Path,
                        help="CSV split manifest of labeled Trident H5 files")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n-components", type=int, default=BEST_N_COMPONENTS)
    parser.add_argument("--C", dest="c_value", type=float, default=BEST_C)
    parser.add_argument("--cache-size", type=float, default=DEFAULT_CACHE_SIZE_MB)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Append val to train. By default, only the train split is fitted.",
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        load_split = lambda split: load_h5_split_samples(args.manifest, split)
        load_dataset = load_labeled_h5_dataset
    else:
        load_split = lambda split: load_split_samples(args.xlsx, args.feature_root, split)
        load_dataset = load_labeled_dataset

    print("Loading training manifest", flush=True)
    train_df = load_split("train")
    fitted_splits = ["train"]
    val_df = load_split("val") if args.manifest is not None or args.include_val else None
    if args.include_val:
        assert val_df is not None
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        fitted_splits.append("val")

    print("Loading test manifest", flush=True)
    test_df = load_split("test")

    print(f"Loading {len(train_df):,} fitting samples", flush=True)
    train_data = load_dataset(train_df)
    val_data = load_dataset(val_df) if val_df is not None and not args.include_val else None
    print(f"Loading {len(test_df):,} test samples", flush=True)
    test_data = load_dataset(test_df)

    print(
        f"Training GPU SVC on {len(train_data.labels):,} tiles with "
        f"{train_data.features.shape[1]:,} features",
        flush=True,
    )
    bundle = train_gpu_pipeline(
        train_data.features,
        train_data.labels,
        n_components=args.n_components,
        c_value=args.c_value,
        cache_size_mb=args.cache_size,
    )

    metadata = base_metadata()
    metadata.update(
        {
            "model_type": "cuml StandardScaler -> PCA -> RBF SVC",
            "training_source": "trident_h5" if args.manifest is not None else "pt_csv",
            "fitted_splits": fitted_splits,
            "evaluation_split": "test",
            "sample_counts": {
                "fit": int(len(train_df)),
                "val": int(len(val_df)) if val_data is not None else 0,
                "test": int(len(test_df)),
            },
            "tile_counts": {
                "fit": int(len(train_data.labels)),
                "val": int(len(val_data.labels)) if val_data is not None else 0,
                "test": int(len(test_data.labels)),
            },
            "class_counts_fit": {
                LABEL_NAMES[class_id]: int((train_data.labels == class_id).sum())
                for class_id in (0, 1)
            },
            "feature_dimension": int(train_data.features.shape[1]),
            "n_components": int(args.n_components),
            "C": float(args.c_value),
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "cache_size_mb": float(args.cache_size),
            "decision_threshold": float(args.threshold),
            "encoded_labels": LABEL_NAMES,
            "raw_label_mapping": RAW_LABEL_MAPPING,
            "gpu": bundle["gpu"],
        }
    )
    if args.manifest is not None:
        metadata["manifest_path"] = str(args.manifest.expanduser().resolve())
    else:
        metadata["xlsx_path"] = str(args.xlsx.expanduser().resolve())
        metadata["feature_root"] = str(args.feature_root.expanduser().resolve())

    model_path = save_bundle(bundle, output_dir / "svc_pipeline.pkl", metadata)
    write_json(metadata, output_dir / "metadata.json")
    print(f"Saved model: {model_path}", flush=True)

    print("Evaluating fitted training tiles", flush=True)
    train_pred, train_scores = predict_gpu(
        bundle,
        train_data.features,
        batch_size=args.prediction_batch_size,
        threshold=args.threshold,
    )
    train_true = train_data.labels.numpy().astype(np.int32, copy=False)
    train_metrics = classification_metrics(train_true, train_pred, train_scores)
    train_metrics["threshold"] = float(args.threshold)
    train_metrics["train_tiles"] = int(len(train_true))
    train_metrics["fitted_splits"] = fitted_splits
    write_json(train_metrics, output_dir / "train_metrics.json")

    if val_data is not None:
        print("Evaluating validation tiles", flush=True)
        val_pred, val_scores = predict_gpu(
            bundle, val_data.features,
            batch_size=args.prediction_batch_size,
            threshold=args.threshold,
        )
        val_true = val_data.labels.numpy().astype(np.int32, copy=False)
        val_metrics = classification_metrics(val_true, val_pred, val_scores)
        val_metrics["threshold"] = float(args.threshold)
        val_metrics["val_tiles"] = int(len(val_true))
        write_json(val_metrics, output_dir / "val_metrics.json")

    print("Evaluating held-out test tiles", flush=True)
    y_pred, scores = predict_gpu(
        bundle,
        test_data.features,
        batch_size=args.prediction_batch_size,
        threshold=args.threshold,
    )
    y_true = test_data.labels.numpy().astype(np.int32, copy=False)
    metrics = classification_metrics(y_true, y_pred, scores)
    metrics["threshold"] = float(args.threshold)
    metrics["test_tiles"] = int(len(y_true))
    write_json(metrics, output_dir / "test_metrics.json")
    save_confusion_matrix(metrics, output_dir / "test_confusion_matrix.png")

    predictions = labeled_predictions_dataframe(
        sample_names=test_data.sample_names,
        coordinates=test_data.coordinates,
        pt_paths=test_data.pt_paths,
        y_true=y_true,
        y_pred=y_pred,
        decision_scores=scores,
        path_column="feature_path" if args.manifest is not None else "pt_path",
    )
    predictions.to_csv(output_dir / "test_tile_predictions.csv", index=False)

    elapsed = time.monotonic() - started
    print(
        f"Complete | train_balanced_accuracy={train_metrics['balanced_accuracy']:.6f} | "
        f"test_balanced_accuracy={metrics['balanced_accuracy']:.6f} | "
        f"elapsed_seconds={elapsed:.1f} | output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

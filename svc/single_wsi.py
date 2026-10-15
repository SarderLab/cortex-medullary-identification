"""TRIDENT UNI-v1 feature extraction, GT labeling, and SVC WSI evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .config import DEFAULT_INFERENCE_BATCH_SIZE, DEFAULT_MODEL_DIR, DEFAULT_THRESHOLD, LABEL_NAMES
from .evaluation import classification_metrics
from .tile_annotations import (
    ANNOTATION_FILES,
    label_tiles,
    load_annotation_geometry,
    resolve_annotation_dir,
    write_tile_annotation_csv,
)


def _attribute(containers: list[Any], names: tuple[str, ...], default: Any = None) -> Any:
    for container in containers:
        if container is None:
            continue
        for name in names:
            if name in container.attrs:
                value = container.attrs[name]
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return value
    return default


def _find_feature_h5(
    processed_dir: Path, wsi_path: Path, mag: float, patch_size: int, overlap_px: int,
) -> Path:
    mag_name = f"{mag:g}x_{patch_size}px_{overlap_px}px_overlap"
    feature_dir = processed_dir / mag_name / "features_uni_v1"
    candidates = [feature_dir / f"{wsi_path.stem}.h5", feature_dir / f"{wsi_path.name}.h5"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(feature_dir.glob("*.h5")) if feature_dir.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot identify TRIDENT UNI-v1 output under {feature_dir}; "
        f"expected {wsi_path.stem}.h5 or {wsi_path.name}.h5"
    )


def load_feature_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray, int, int | None, int | None, str]:
    """Read TRIDENT's native HDF5 datasets directly without creating a PT copy."""
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle or "coords" not in handle:
            raise ValueError(f"TRIDENT feature H5 must contain 'features' and 'coords': {h5_path}")
        features = np.asarray(handle["features"], dtype=np.float32)
        coordinates_raw = np.asarray(handle["coords"])
        if not np.isfinite(coordinates_raw).all():
            raise ValueError(f"Non-finite coordinates in {h5_path}")
        coordinates = np.asarray(coordinates_raw, dtype=np.int64)
        containers = [handle, handle["coords"]]
        tile_size_value = _attribute(containers, ("patch_size_level0", "tile_size"))
        if tile_size_value is None:
            raise ValueError(
                f"TRIDENT H5 has no patch_size_level0 metadata: {h5_path}; "
                "refusing to assume the level-zero tile size"
            )
        tile_size = int(tile_size_value)
        width_value = _attribute(containers, ("level0_width", "wsi_width", "width"))
        height_value = _attribute(containers, ("level0_height", "wsi_height", "height"))
        encoder_name = str(_attribute([handle], ("encoder",), "uni_v1"))

    if features.ndim != 2 or not len(features):
        raise ValueError(f"Expected nonempty [tiles, features] matrix in {h5_path}; got {features.shape}")
    if coordinates.shape != (len(features), 2):
        raise ValueError(f"Expected coordinates shape {(len(features), 2)} in {h5_path}; got {coordinates.shape}")
    if not np.isfinite(features).all() or not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite feature or coordinate values in {h5_path}")
    if tile_size <= 0:
        raise ValueError(f"Invalid patch_size_level0={tile_size} in {h5_path}")

    return (
        features,
        coordinates,
        tile_size,
        int(width_value) if width_value is not None else None,
        int(height_value) if height_value is not None else None,
        encoder_name,
    )


def _link_native_embedding(h5_path: Path, processed_dir: Path, sample_name: str) -> Path:
    """Expose the unmodified TRIDENT HDF5 file in the requested embeddings folder."""
    embedding_dir = processed_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    destination = embedding_dir / f"{sample_name}.h5"
    if destination.exists():
        try:
            if destination.samefile(h5_path):
                return destination
        except OSError:
            pass
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    os.symlink(os.path.relpath(h5_path, destination.parent), temporary)
    os.replace(temporary, destination)
    return destination


def _validate_physical_tile_size(
    tile_size: int,
    *,
    mpp_x: float | None,
    mpp_y: float | None,
    expected_microns: float,
    maximum_error_percent: float,
) -> tuple[float, float] | None:
    """Check TRIDENT's level-zero tile edge against the planned physical field."""
    if mpp_x is None or mpp_y is None:
        return None
    if mpp_x <= 0 or mpp_y <= 0 or expected_microns <= 0 or maximum_error_percent < 0:
        raise ValueError("MPP, expected field of view, and tolerance must be positive")
    measured = (tile_size * mpp_x, tile_size * mpp_y)
    errors = tuple(abs(value - expected_microns) / expected_microns * 100.0 for value in measured)
    if any(error > maximum_error_percent for error in errors):
        raise ValueError(
            "TRIDENT tile physical size does not match the requested field of view: "
            f"HDF5 patch_size_level0={tile_size}, measured={measured[0]:.3f}x{measured[1]:.3f} um, "
            f"target={expected_microns:.3f} um, error={errors[0]:.2f}%x{errors[1]:.2f}% "
            f"(limit {maximum_error_percent:.2f}%). Check WSI MPP/objective metadata or override it."
        )
    return measured


def _save_metrics_txt(
    processed_dir: Path,
    *,
    sample_name: str,
    wsi_path: Path,
    annotation_dir: Path,
    feature_h5: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Path:
    """Write date-named, human-readable metrics and a labeled confusion matrix."""
    metrics_dir = processed_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone()
    destination = metrics_dir / f"{generated_at.date().isoformat()}.txt"
    temporary = destination.with_suffix(".txt.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(f"SVC whole-slide evaluation: {sample_name}\n")
        stream.write(f"Generated: {generated_at.isoformat(timespec='seconds')}\n")
        stream.write(f"WSI: {wsi_path}\n")
        stream.write(f"Annotations: {annotation_dir}\n")
        stream.write(f"Native UNI HDF5: {feature_h5}\n")
        stream.write(f"Decision threshold: {threshold:.6g}\n")
        stream.write(f"Labeled tiles evaluated: {len(y_true):,}\n\n")
        if not len(y_true):
            stream.write("No annotated cortex or medullary tiles overlap the TRIDENT tile grid.\n")
            stream.write("Overall accuracy: N/A\nBalanced accuracy: N/A\n\n")
            stream.write("Per-class metrics (no labeled tiles):\n")
            for class_name in ("cortex", "medullary"):
                stream.write(
                    f"  {class_name}: accuracy=N/A, balanced_accuracy=N/A, AUC=N/A, "
                    "sensitivity=N/A, specificity=N/A, support=0\n"
                )
            stream.write("\nConfusion matrix (rows=true, columns=predicted; cortex, medullary):\n")
            stream.write("                 cortex  medullary\n  cortex              0          0\n  medullary           0          0\n")
        else:
            report = classification_metrics(y_true, y_pred, scores)
            stream.write(f"Overall accuracy: {report['accuracy']:.6f}\n")
            stream.write(f"Balanced accuracy: {report['balanced_accuracy']:.6f}\n\n")
            stream.write("Per-class metrics (accuracy and AUC are one-vs-rest):\n")
            for encoded, class_name in ((0, "cortex"), (1, "medullary")):
                row = report["per_class"][LABEL_NAMES[encoded]]
                auc = "N/A" if row["auc"] is None else f"{row['auc']:.6f}"
                sensitivity = "N/A" if row["sensitivity"] is None else f"{row['sensitivity']:.6f}"
                specificity = "N/A" if row["specificity"] is None else f"{row['specificity']:.6f}"
                balanced = (
                    "N/A" if row["sensitivity"] is None or row["specificity"] is None
                    else f"{(row['sensitivity'] + row['specificity']) / 2.0:.6f}"
                )
                stream.write(
                    f"  {class_name}: accuracy={row['accuracy']:.6f}, "
                    f"balanced_accuracy={balanced}, AUC={auc}, sensitivity={sensitivity}, "
                    f"specificity={specificity}, support={row['support']:,}\n"
                )
            matrix = report["confusion_matrix"]
            stream.write("\nConfusion matrix (rows=true, columns=predicted):\n")
            stream.write("                 cortex  medullary\n")
            stream.write(f"  cortex         {matrix[0][0]:7d}  {matrix[0][1]:9d}\n")
            stream.write(f"  medullary      {matrix[1][0]:7d}  {matrix[1][1]:9d}\n")
    os.replace(temporary, destination)
    return destination


def _rectangle_feature(x: float, y: float, width: float, height: float, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]]],
        },
        "properties": properties,
    }


def _write_feature_collection(path: Path, features: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": "FeatureCollection", "properties": metadata, "features": features}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _prediction_features(coordinates: np.ndarray, scores: np.ndarray, labels: np.ndarray, tile_size: int) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for index, ((x, y), score, label) in enumerate(zip(coordinates, scores, labels)):
        class_id = int(label)
        class_name = "cortex" if class_id == 0 else "medullary"
        features.append(_rectangle_feature(
            int(x), int(y), tile_size, tile_size,
            {
                "tile_index": index,
                "tile_x": int(x),
                "tile_y": int(y),
                "class_id": class_id,
                "class": class_name,
                "label": class_name,
                "decision_score": float(score),
            },
        ))
    return features


def _gaussian_features(
    coordinates: np.ndarray,
    scores: np.ndarray,
    *,
    tile_size: int,
    threshold: float,
    sigma_tile_widths: float,
    cell_size: int,
) -> list[dict[str, Any]]:
    """Make a sparse Gaussian score map over covered tile footprints."""
    sigma = max(0.05, min(0.75, float(sigma_tile_widths))) * tile_size
    cells: set[tuple[int, int]] = set()
    for x_value, y_value in coordinates:
        x, y = int(x_value), int(y_value)
        left = math.floor(x / cell_size)
        top = math.floor(y / cell_size)
        right = math.ceil((x + tile_size) / cell_size)
        bottom = math.ceil((y + tile_size) / cell_size)
        for gy in range(top, bottom):
            cy = (gy + 0.5) * cell_size
            if not y <= cy < y + tile_size:
                continue
            for gx in range(left, right):
                cx = (gx + 0.5) * cell_size
                if x <= cx < x + tile_size:
                    cells.add((gx, gy))

    centers = np.asarray(coordinates, dtype=np.float64) + tile_size / 2.0
    tree = cKDTree(centers)
    ordered_cells = sorted(cells, key=lambda cell: (cell[1], cell[0]))
    features: list[dict[str, Any]] = []
    radius = 3.0 * sigma
    for start in range(0, len(ordered_cells), 8192):
        group = ordered_cells[start:start + 8192]
        points = np.asarray([[(gx + 0.5) * cell_size, (gy + 0.5) * cell_size] for gx, gy in group])
        neighbors = tree.query_ball_point(points, r=radius)
        for (gx, gy), point, neighbor_ids in zip(group, points, neighbors):
            if neighbor_ids:
                delta = centers[neighbor_ids] - point
                weights = np.exp(-np.einsum("ij,ij->i", delta, delta) / (2.0 * sigma * sigma))
                refined_score = float(np.dot(weights, scores[neighbor_ids]) / weights.sum())
            else:
                nearest = int(tree.query(point, k=1)[1])
                refined_score = float(scores[nearest])
            class_id = int(refined_score >= threshold)
            features.append(_rectangle_feature(
                gx * cell_size, gy * cell_size, cell_size, cell_size,
                {
                    "class_id": class_id,
                    "class": "cortex" if class_id == 0 else "medullary",
                    "label": "cortex" if class_id == 0 else "medullary",
                    "refined_decision_score": refined_score,
                    "threshold": float(threshold),
                    "aggregation": "gaussian",
                    "sigma_tile_widths": float(sigma_tile_widths),
                    "sigma_level0_pixels": float(sigma),
                    "source_tile_size_level0": int(tile_size),
                },
            ))
    return features


def _find_and_canonicalize_contour(processed_dir: Path, sample_name: str) -> None:
    contour_dir = processed_dir / "contours_geojson"
    canonical = contour_dir / f"{sample_name}.geojson"
    if canonical.is_file():
        return
    candidates = sorted(contour_dir.glob("*.geojson")) if contour_dir.is_dir() else []
    if len(candidates) == 1:
        import shutil
        shutil.copy2(candidates[0], canonical)
    else:
        raise FileNotFoundError(f"TRIDENT did not create a unique tissue contour in {contour_dir}")


def run_trident(args: argparse.Namespace, wsi_path: Path, processed_dir: Path) -> None:
    command = shlex.split(args.trident_command)
    if not command:
        raise ValueError("TRIDENT command cannot be empty")
    command += [
        "single", "--", "--slide_path", str(wsi_path), "--job_dir", str(processed_dir),
        "--segmenter", args.segmenter,
        "--patch_encoder", "uni_v1",
        "--mag", str(args.magnification),
        "--patch_size", str(args.patch_size),
        "--overlap", str(args.overlap_px),
        "--patch_encoder_img_size", str(args.encoder_img_size),
        "--gpu", str(args.gpu),
        "--batch_size", str(args.trident_batch_size),
    ]
    if args.encoder_checkpoint:
        command.extend(["--patch_encoder_ckpt_path", str(args.encoder_checkpoint)])
    print("Running TRIDENT: " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, default=None,
                        help="Optional sample annotation folder/root; omit for inference-only output.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("SVC_MODEL_PATH", str(DEFAULT_MODEL_DIR / "svc_pipeline.pkl"))),
    )
    parser.add_argument("--trident-command", default=os.environ.get("TRIDENT_COMMAND", "trident"))
    parser.add_argument("--segmenter", default=os.environ.get("TRIDENT_SEGMENTER", "hest"))
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--magnification", type=float, default=40.0)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap-px", type=int, default=0)
    parser.add_argument("--encoder-img-size", type=int, default=512)
    parser.add_argument("--mpp-x", type=float, default=None)
    parser.add_argument("--mpp-y", type=float, default=None)
    parser.add_argument("--expected-field-of-view-um", type=float, default=128.0)
    parser.add_argument("--max-field-error-percent", type=float, default=10.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trident-batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--gaussian-sigma", type=float, default=0.25)
    parser.add_argument("--refined-cell-size", type=int, default=128)
    parser.add_argument("--skip-trident", action="store_true", help="Reuse an existing TRIDENT run directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wsi_path = args.wsi.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not wsi_path.is_file():
        raise FileNotFoundError(f"WSI does not exist: {wsi_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained SVC model does not exist: {model_path}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_trident:
        run_trident(args, wsi_path, processed_dir)

    sample_name = wsi_path.stem
    _find_and_canonicalize_contour(processed_dir, sample_name)
    h5_path = _find_feature_h5(
        processed_dir, wsi_path, args.magnification, args.patch_size, args.overlap_px,
    )
    features, coordinates, tile_size, width, height, encoder_name = load_feature_h5(h5_path)
    if encoder_name.lower() not in ("uni_v1", "uni", "unknown", ""):
        raise ValueError(f"Expected UNI-v1 features, but HDF5 encoder is {encoder_name!r}: {h5_path}")
    if args.encoder_img_size != 512:
        raise ValueError(f"UNI input must be resized to 512x512, got {args.encoder_img_size}")
    measured_fov = _validate_physical_tile_size(
        tile_size,
        mpp_x=args.mpp_x,
        mpp_y=args.mpp_y,
        expected_microns=args.expected_field_of_view_um,
        maximum_error_percent=args.max_field_error_percent,
    )
    native_embedding = _link_native_embedding(h5_path, processed_dir, sample_name)

    annotation_dir = None
    raw_labels = None
    cortex_overlap = medullary_overlap = None
    tile_annotation_csv = None
    if args.annotation_dir is not None:
        annotation_dir = resolve_annotation_dir(args.annotation_dir, sample_name)
        cortex_geometry = load_annotation_geometry(annotation_dir / ANNOTATION_FILES["cortex"])
        medullary_geometry = load_annotation_geometry(annotation_dir / ANNOTATION_FILES["medullary"])
        raw_labels, cortex_overlap, medullary_overlap = label_tiles(
            coordinates, tile_size, cortex_geometry, medullary_geometry,
        )
        tile_annotation_csv = processed_dir / "tile_annotations" / f"{sample_name}.csv"
        # Persist the annotation-to-tile mapping before inference. Predictions
        # and margins are filled into the same CSV after the SVC has run.
        write_tile_annotation_csv(
            tile_annotation_csv, coordinates, tile_size, raw_labels,
            cortex_overlap, medullary_overlap,
        )

    # Keep GPU-only imports out of module initialization so annotation and
    # HDF5 helpers remain testable on CPU-only hosts.
    import torch

    from .gpu_model import load_bundle, predict_gpu

    bundle = load_bundle(model_path)
    expected_dim = int(bundle["metadata"].get("feature_dimension", 0))
    if expected_dim and features.shape[1] != expected_dim:
        raise ValueError(f"SVC expects {expected_dim} features, but {h5_path} contains {features.shape[1]}")
    predictions, scores = predict_gpu(
        bundle,
        torch.from_numpy(np.ascontiguousarray(features)),
        batch_size=args.inference_batch_size,
        threshold=args.threshold,
    )
    if tile_annotation_csv is not None:
        write_tile_annotation_csv(
            tile_annotation_csv, coordinates, tile_size, raw_labels,
            cortex_overlap, medullary_overlap, predictions, scores,
        )

    metrics_path = None
    eval_tile_count = 0
    if annotation_dir is not None:
        valid_gt = np.isin(raw_labels, (1, 2))
        eval_tile_count = int(valid_gt.sum())
        y_true = (raw_labels[valid_gt] == 2).astype(np.int32)
        metrics_path = _save_metrics_txt(
            processed_dir,
            sample_name=sample_name,
            wsi_path=wsi_path,
            annotation_dir=annotation_dir,
            feature_h5=h5_path,
            y_true=y_true,
            y_pred=predictions[valid_gt],
            scores=scores[valid_gt],
            threshold=args.threshold,
        )

    labels = _prediction_features(coordinates, scores, predictions, tile_size)
    metadata = {
        "sample_name": sample_name,
        "wsi_path": str(wsi_path),
        "feature_h5": str(h5_path),
        "embedding_h5_link": str(native_embedding),
        "encoder": "uni_v1",
        "encoder_input_size": [512, 512],
        "trident_magnification": float(args.magnification),
        "trident_patch_size": int(args.patch_size),
        "wsi_mpp_level0": [args.mpp_x, args.mpp_y],
        "model_path": str(model_path),
        "tile_count": int(len(predictions)),
        "tile_size_level0": int(tile_size),
        "tile_field_of_view_um": list(measured_fov) if measured_fov else None,
        "target_field_of_view_um": float(args.expected_field_of_view_um),
        "tile_field_error_percent": (
            [round((value / args.expected_field_of_view_um - 1.0) * 100.0, 3) for value in measured_fov]
            if measured_fov else None
        ),
        "overlap_pixels_at_requested_magnification": int(args.overlap_px),
        "threshold": float(args.threshold),
        "classes": {"0": "cortex", "1": "medullary"},
        "score_definition": "signed SVC decision margin; positive scores favor medullary",
        "coordinate_space": "level-zero WSI pixels",
        "annotation_dir": str(annotation_dir) if annotation_dir else None,
        "labeled_tile_count": eval_tile_count,
    }
    heatmap_dir = processed_dir / "heatmap_json"
    _write_feature_collection(heatmap_dir / f"{sample_name}_heatmap.json", labels, {**metadata, "kind": "tile heatmap"})
    _write_feature_collection(heatmap_dir / f"{sample_name}_mask.json", labels, {**metadata, "kind": "tile label predictions"})

    cell_size = max(32, min(int(args.refined_cell_size), tile_size))
    refined = _gaussian_features(
        coordinates,
        scores,
        tile_size=tile_size,
        threshold=args.threshold,
        sigma_tile_widths=args.gaussian_sigma,
        cell_size=cell_size,
    )
    _write_feature_collection(
        heatmap_dir / f"{sample_name}_refined_heatmap.json",
        refined,
        {
            **metadata,
            "kind": "Gaussian-refined heatmap",
            "aggregation": "Gaussian-weighted mean of signed tile-center decision scores",
            "sigma_tile_widths": float(args.gaussian_sigma),
            "sigma_level0_pixels": float(args.gaussian_sigma * tile_size),
            "cell_size_level0": int(cell_size),
            "width_level0": width,
            "height_level0": height,
        },
    )

    summary = {
        **metadata,
        "contour_geojson": str(processed_dir / "contours_geojson" / f"{sample_name}.geojson"),
        "cortical_tiles": int(np.sum(predictions == 0)),
        "medullary_tiles": int(np.sum(predictions == 1)),
        "tile_annotation_csv": str(tile_annotation_csv) if tile_annotation_csv else None,
        "metrics_txt": str(metrics_path) if metrics_path else None,
        "heatmap_json": str(heatmap_dir / f"{sample_name}_heatmap.json"),
        "refined_heatmap_json": str(heatmap_dir / f"{sample_name}_refined_heatmap.json"),
        "tile_label_predictions_json": str(heatmap_dir / f"{sample_name}_mask.json"),
    }
    summary_path = processed_dir / "inference_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    with temporary_summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_summary, summary_path)
    print(
        f"Complete: {sample_name} | tiles={len(predictions):,} "
        f"cortex={summary['cortical_tiles']:,} medullary={summary['medullary_tiles']:,} "
        f"GT tiles={eval_tile_count:,} outputs={heatmap_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

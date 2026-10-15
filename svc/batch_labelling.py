"""Add GT tile labels to TRIDENT patch feature H5 files, with a CSV run report."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

from .tile_annotations import ANNOTATION_FILES, load_annotation_geometry


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=Path,
                        help="Root containing one folder per WSI sample")
    parser.add_argument("--job-dir", required=True, type=Path,
                        help="TRIDENT --job_dir containing patch feature H5 files")
    parser.add_argument("--features-dir", type=Path,
                        help="Specific TRIDENT features_<encoder> folder, if more than one exists")
    parser.add_argument("--report", type=Path,
                        help="CSV report path (default: JOB_DIR/batch_labelling_report.csv)")
    return parser.parse_args()


def _feature_h5(sample: str, job_dir: Path, features_dir: Path | None) -> Path | None:
    if features_dir is not None:
        candidate = features_dir / f"{sample}.h5"
        return candidate if candidate.is_file() else None
    candidates = sorted(
        feature_dir / f"{sample}.h5"
        for feature_dir in job_dir.rglob("features_*")
        if feature_dir.is_dir() and (feature_dir / f"{sample}.h5").is_file()
    )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple feature H5 files for {sample}: {', '.join(map(str, candidates))}; "
            "pass --features-dir"
        )
    return candidates[0] if candidates else None


def _overlap_areas(coords: np.ndarray, tile_size: int, geometry: Any | None) -> np.ndarray:
    areas = np.zeros(len(coords), dtype=np.float64)
    if geometry is None or geometry.is_empty:
        return areas

    # Shapely 2 computes intersections in batches. Shapely 1 uses a prepared
    # polygon to avoid constructing most intersections outside the GT bounds.
    try:
        import shapely

        if int(shapely.__version__.split(".", 1)[0]) >= 2:
            for start in range(0, len(coords), 2048):
                xy = coords[start:start + 2048]
                tiles = shapely.box(xy[:, 0], xy[:, 1],
                                    xy[:, 0] + tile_size, xy[:, 1] + tile_size)
                intersects = shapely.intersects(tiles, geometry)
                if np.any(intersects):
                    indexes = np.flatnonzero(intersects)
                    areas[start + indexes] = shapely.area(
                        shapely.intersection(tiles[indexes], geometry)
                    )
            return areas
    except (ImportError, AttributeError, ValueError):
        pass

    prepared = prep(geometry)
    for index, (x, y) in enumerate(coords):
        tile = box(float(x), float(y), float(x + tile_size), float(y + tile_size))
        if prepared.intersects(tile):
            areas[index] = geometry.intersection(tile).area
    return areas


def _labels(coords: np.ndarray, tile_size: int, cortex: Any | None,
            medullary: Any | None) -> np.ndarray:
    cortex_area = _overlap_areas(coords, tile_size, cortex)
    medullary_area = _overlap_areas(coords, tile_size, medullary)
    labels = np.zeros(len(coords), dtype=np.int8)
    annotated = (cortex_area > 0) | (medullary_area > 0)
    labels[annotated] = 2
    labels[annotated & (cortex_area > medullary_area)] = 1
    return labels


def _write_labels(handle: h5py.File, labels: np.ndarray) -> None:
    pending = "_labels_pending"
    previous = "_labels_previous"
    if pending in handle or previous in handle:
        raise ValueError("Incomplete prior label update exists in H5; inspect it before retrying")
    if "labels" in handle and not isinstance(handle["labels"], h5py.Dataset):
        raise ValueError("H5 path 'labels' already exists and is not a dataset")

    dataset = handle.create_dataset(pending, data=labels, compression="gzip")
    dataset.attrs["0"] = "unlabeled"
    dataset.attrs["1"] = "cortex"
    dataset.attrs["2"] = "medullary"
    dataset.attrs["rule"] = "larger annotated overlap area; ties go to medullary"
    handle.flush()
    had_previous = "labels" in handle
    if had_previous:
        handle.move("labels", previous)
    handle.move(pending, "labels")
    handle.flush()
    if had_previous:
        del handle[previous]


def _label_h5(path: Path, cortex: Any | None, medullary: Any | None) -> dict[str, int]:
    with h5py.File(path, "r+") as handle:
        if "features" not in handle or "coords" not in handle:
            raise ValueError("H5 must contain 'features' and 'coords' datasets")
        feature_count = len(handle["features"])
        coords = np.asarray(handle["coords"][:])
        if coords.shape != (feature_count, 2) or not np.isfinite(coords).all():
            raise ValueError("coords must contain one finite (x, y) pair per feature row")
        if not np.equal(coords, np.floor(coords)).all():
            raise ValueError("coords must be integer level-zero pixel origins")
        coords = coords.astype(np.int64)
        tile_size = handle["coords"].attrs.get(
            "patch_size_level0", handle.attrs.get("patch_size_level0")
        )
        if tile_size is None or int(tile_size) <= 0:
            raise ValueError("H5 coords lack a positive patch_size_level0 attribute")
        labels = _labels(coords, int(tile_size), cortex, medullary)
        _write_labels(handle, labels)
    counts = np.bincount(labels, minlength=3)
    return {
        "total_tiles": feature_count,
        "labelled_tiles": int(counts[1] + counts[2]),
        "cortex_tiles": int(counts[1]),
        "medullary_tiles": int(counts[2]),
        "unlabeled_tiles": int(counts[0]),
    }


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    fields = ["sample", "status", "h5_path", "total_tiles", "labelled_tiles",
              "cortex_tiles", "medullary_tiles", "unlabeled_tiles", "detail"]
    with pending.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(pending, path)


def main() -> int:
    args = _arguments()
    annotation_dir = args.annotation_dir.expanduser().resolve()
    job_dir = args.job_dir.expanduser().resolve()
    features_dir = args.features_dir.expanduser().resolve() if args.features_dir else None
    report = args.report.expanduser().resolve() if args.report else job_dir / "batch_labelling_report.csv"
    if not annotation_dir.is_dir():
        raise SystemExit(f"Annotation directory not found: {annotation_dir}")
    if not job_dir.is_dir():
        raise SystemExit(f"TRIDENT job directory not found: {job_dir}")
    if features_dir is not None and not features_dir.is_dir():
        raise SystemExit(f"Feature directory not found: {features_dir}")

    rows: list[dict[str, Any]] = []
    failures = 0
    for sample_dir in sorted(path for path in annotation_dir.iterdir() if path.is_dir()):
        row: dict[str, Any] = {"sample": sample_dir.name, "status": "", "h5_path": "",
                               "total_tiles": 0, "labelled_tiles": 0, "cortex_tiles": 0,
                               "medullary_tiles": 0, "unlabeled_tiles": 0, "detail": ""}
        gt_paths = {name: sample_dir / file_name
                    for name, file_name in ANNOTATION_FILES.items()}
        if not any(path.is_file() for path in gt_paths.values()):
            row["status"] = "skipped_no_json"
        else:
            try:
                h5_path = _feature_h5(sample_dir.name, job_dir, features_dir)
                if h5_path is None:
                    row["status"] = "missing_h5"
                    row["detail"] = "No matching TRIDENT patch feature H5"
                else:
                    row["h5_path"] = str(h5_path)
                    cortex = load_annotation_geometry(gt_paths["cortex"])
                    medullary = load_annotation_geometry(gt_paths["medullary"])
                    row.update(_label_h5(h5_path, cortex, medullary))
                    row["status"] = "processed"
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["status"] = "error"
                row["detail"] = str(error)
        if row["status"] in {"error", "missing_h5"}:
            failures += 1
        rows.append(row)
        print(f"{row['sample']}: {row['status']} (cortex={row['cortex_tiles']}, "
              f"medullary={row['medullary_tiles']}, total={row['total_tiles']})", flush=True)

    _write_report(report, rows)
    print(f"Report: {report} ({len(rows)} sample folders; {failures} failures)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

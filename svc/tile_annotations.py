"""Rasterize cortex/medullary polygon annotations onto TRIDENT tile coordinates."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from shapely.validation import explain_validity


ANNOTATION_FILES = {
    "cortex": "cortical_gt.json",
    "medullary": "medullary_gt.json",
}


def resolve_annotation_dir(annotation_path: Path, sample_name: str) -> Path:
    """Accept either a sample folder or a root containing one folder per slide."""
    annotation_path = annotation_path.expanduser().resolve()
    if any((annotation_path / name).is_file() for name in ANNOTATION_FILES.values()):
        return annotation_path
    sample_dir = annotation_path / sample_name
    if sample_dir.is_dir():
        return sample_dir
    raise FileNotFoundError(
        f"No {ANNOTATION_FILES['cortex']} or {ANNOTATION_FILES['medullary']} found in "
        f"{annotation_path} or {sample_dir}"
    )


def _polygon_from_histomics(element: dict[str, Any]) -> Any | None:
    if element.get("type") != "polyline" or element.get("closed") is not True:
        return None
    points = element.get("points", [])
    if len(points) < 3:
        return None
    try:
        polygon = Polygon([[float(point[0]), float(point[1])] for point in points])
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("Histomics polygon points must be [x,y] or [x,y,z] arrays") from error
    return polygon


def _geometries_from_document(document: Any) -> list[Any]:
    if not isinstance(document, dict):
        raise ValueError("Annotation JSON must contain an object")
    if isinstance(document.get("document"), dict):
        document = document["document"]

    kind = document.get("type")
    if kind == "FeatureCollection":
        geometries = []
        for feature in document.get("features", []):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                continue
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict) or geometry.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            geometries.append(shape(geometry))
        return geometries
    if kind == "Feature":
        geometry = document.get("geometry")
        if isinstance(geometry, dict) and geometry.get("type") in ("Polygon", "MultiPolygon"):
            return [shape(geometry)]
        return []
    if kind in ("Polygon", "MultiPolygon"):
        return [shape(document)]

    annotation = document.get("annotation", document)
    elements = annotation.get("elements") if isinstance(annotation, dict) else None
    if not isinstance(elements, list):
        raise ValueError(
            "Supported annotation schemas are GeoJSON Polygon/MultiPolygon or "
            "closed Histomics polyline elements"
        )
    return [polygon for element in elements if isinstance(element, dict)
            if (polygon := _polygon_from_histomics(element)) is not None]


def load_annotation_geometry(path: Path) -> Any | None:
    """Load and union polygons from one saved GT JSON; a missing class is empty."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    geometries = _geometries_from_document(payload)
    valid = []
    for geometry in geometries:
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            raise ValueError(f"Invalid polygon in {path}: {explain_validity(geometry)}")
        valid.append(geometry)
    return unary_union(valid) if valid else None


def _positive_area_overlaps(coordinates: np.ndarray, tile_size: int, geometry: Any | None) -> np.ndarray:
    """Return whether each tile intersects a polygon interior with positive area."""
    if geometry is None or geometry.is_empty:
        return np.zeros(len(coordinates), dtype=bool)

    try:
        import shapely

        # Shapely 2 runs the predicates in vectorized GEOS batches.
        if int(shapely.__version__.split(".", 1)[0]) >= 2:
            result = np.zeros(len(coordinates), dtype=bool)
            for start in range(0, len(coordinates), 4096):
                xy = coordinates[start:start + 4096]
                tiles = shapely.box(xy[:, 0], xy[:, 1], xy[:, 0] + tile_size, xy[:, 1] + tile_size)
                result[start:start + len(xy)] = shapely.relate_pattern(tiles, geometry, "T********")
            return result
    except (ImportError, AttributeError, ValueError):
        pass

    # Compatible path for Shapely 1.x installations.
    from shapely.prepared import prep

    prepared = prep(geometry)
    result = np.zeros(len(coordinates), dtype=bool)
    for index, (x_value, y_value) in enumerate(coordinates):
        tile = box(float(x_value), float(y_value), float(x_value + tile_size), float(y_value + tile_size))
        if prepared.intersects(tile):
            result[index] = geometry.intersection(tile).area > 0.0
    return result


def label_tiles(
    coordinates: np.ndarray,
    tile_size: int,
    cortex_geometry: Any | None,
    medullary_geometry: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw labels and per-class overlap flags.

    Raw labels match the training CSV convention: 1=cortex, 2=medullary,
    0=background/unlabeled. Medullary GT takes precedence where polygons
    overlap, matching the existing annotation workflow.
    """
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must have shape [tiles, 2], got {coordinates.shape}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    cortex_overlap = _positive_area_overlaps(coordinates, tile_size, cortex_geometry)
    medullary_overlap = _positive_area_overlaps(coordinates, tile_size, medullary_geometry)
    labels = np.zeros(len(coordinates), dtype=np.int8)
    labels[cortex_overlap] = 1
    labels[medullary_overlap] = 2
    return labels, cortex_overlap, medullary_overlap


def write_tile_annotation_csv(
    path: Path,
    coordinates: np.ndarray,
    tile_size: int,
    labels: np.ndarray,
    cortex_overlap: np.ndarray,
    medullary_overlap: np.ndarray,
    predictions: np.ndarray | None = None,
    decision_scores: np.ndarray | None = None,
) -> Path:
    """Write one row per HDF5 feature/coordinate pair in original row order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    names = {0: "background", 1: "cortex", 2: "medullary"}
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "tile_index", "x", "y", "tile_size_level0", "label", "label_name",
            "cortex_overlap", "medullary_overlap", "evaluation_included",
            "predicted_label", "decision_score",
        ])
        for index, ((x, y), label) in enumerate(zip(coordinates, labels)):
            prediction = ""
            score = ""
            if predictions is not None:
                prediction = "cortex" if int(predictions[index]) == 0 else "medullary"
            if decision_scores is not None:
                score = f"{float(decision_scores[index]):.9g}"
            writer.writerow([
                index, int(x), int(y), int(tile_size), int(label), names[int(label)],
                int(bool(cortex_overlap[index])), int(bool(medullary_overlap[index])),
                int(int(label) in (1, 2)), prediction, score,
            ])
    os.replace(temporary, path)
    return path

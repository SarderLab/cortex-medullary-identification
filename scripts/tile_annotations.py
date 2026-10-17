"""Load cortex and medullary polygon annotations for TRIDENT tile labeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.validation import explain_validity


ANNOTATION_FILES = {
    "cortex": "cortical_gt.json",
    "medullary": "medullary_gt.json",
}


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

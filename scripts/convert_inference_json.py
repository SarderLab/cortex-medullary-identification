"""Convert legacy inference cell maps to DSA annotation JSON without GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .h5_svm import save_json
from .inference import map_payload


LAYERS = ("refined_heatmap", "heatmap", "mask")


def legacy_to_annotation(document: dict, layer: str) -> dict | list[dict]:
    """Rebuild the sparse source grid from a legacy cell list."""
    cells = document["cells"]
    if not cells:
        raise ValueError("Legacy map has no cells")
    stride = int(document["cell_size_level0"])
    origin = np.asarray(document["origin_level0"], dtype=np.int64)
    if stride <= 0 or origin.shape != (2,):
        raise ValueError("Invalid legacy map origin or stride")
    coords = np.asarray([cell[:2] for cell in cells], dtype=np.int64)
    if np.any((coords - origin) < 0) or np.any((coords - origin) % stride):
        raise ValueError("Legacy cells do not lie on the source grid")
    xy = (coords - origin) // stride
    width, height = xy.max(axis=0) + 1
    scores = np.zeros((height, width), dtype=np.float32)
    seen = np.zeros((height, width), dtype=bool)
    mask = layer == "mask"
    for cell, (x, y) in zip(cells, xy):
        if seen[y, x] or len(cell) != (3 if mask else 5) or cell[-1] not in (0, 1):
            raise ValueError("Invalid or duplicate legacy cell")
        seen[y, x] = True
        if mask:
            scores[y, x] = 1 if cell[-1] == 1 else -1
        else:
            probability = float(cell[3])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid legacy medullary probability")
            probability = np.clip(probability, 1e-6, 1 - 1e-6)
            score = np.log(probability / (1 - probability))
            scores[y, x] = score if (cell[-1] == 1 or score < 0) else -1e-6
    return map_payload(document["sample_name"], document["tile_mode"], origin, stride,
                       scores, seen, mask=mask, layer=layer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Existing inference_jsons directory containing legacy cell maps")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Separate destination for Athena annotation JSONs")
    args = parser.parse_args()
    source_root = args.input_dir.expanduser().resolve()
    destination_root = args.output_dir.expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"Input directory does not exist: {source_root}")
    if destination_root == source_root or source_root in destination_root.parents:
        parser.error("--output-dir must be separate from and outside --input-dir")
    converted = skipped = 0
    for source in sorted(source_root.rglob("*.json")):
        layer = next((name for name in LAYERS if source.name.endswith(f"_{name}.json")), None)
        if layer is None:
            continue
        document = json.loads(source.read_text(encoding="utf-8"))
        if "elements" in document:
            skipped += 1
            continue
        destination = destination_root / source.relative_to(source_root)
        if destination.exists():
            skipped += 1
            continue
        temporary = destination.with_name(f".{destination.name}.tmp")
        save_json(temporary, legacy_to_annotation(document, layer))
        temporary.replace(destination)
        converted += 1
        if converted % 100 == 0:
            print(f"Converted {converted} legacy maps", flush=True)
    print(f"Converted {converted} legacy maps to {destination_root}; skipped {skipped} existing or native files")


if __name__ == "__main__":
    main()

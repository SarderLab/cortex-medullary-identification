"""Draw level-zero ground-truth polygons on a TRIDENT WSI thumbnail."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _polygons(geometry: Any | None):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    else:
        raise ValueError(f"Expected polygon GT, got {geometry.geom_type}")


def save_gt_visualization(
    job_dir: Path,
    sample: str,
    *,
    dimensions: tuple[int, int],
    cortex: Any | None,
    medullary: Any | None,
    cortex_tiles: int,
    medullary_tiles: int,
) -> Path:
    """Use the plain TRIDENT thumbnail as the WSI image beneath GT polygons."""
    thumbnail_path = job_dir / "thumbnails" / f"{sample}.jpg"
    if not thumbnail_path.is_file():
        raise FileNotFoundError(f"WSI thumbnail not found: {thumbnail_path}")
    level0_width, level0_height = dimensions
    if level0_width <= 0 or level0_height <= 0:
        raise ValueError("H5 coordinates need positive level0_width and level0_height attributes")

    with Image.open(thumbnail_path) as source:
        image = source.convert("RGBA")
    scale_x = image.width / level0_width
    scale_y = image.height / level0_height

    for geometry, color in ((cortex, (0, 102, 255)), (medullary, (255, 0, 0))):
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        outlines = []
        for polygon in _polygons(geometry):
            exterior = [(round(x * scale_x), round(y * scale_y)) for x, y in polygon.exterior.coords]
            mask_draw.polygon(exterior, fill=100)
            outlines.append(exterior)
            for ring in polygon.interiors:
                interior = [(round(x * scale_x), round(y * scale_y)) for x, y in ring.coords]
                mask_draw.polygon(interior, fill=0)
                outlines.append(interior)
        image.paste(Image.new("RGBA", image.size, (*color, 255)), (0, 0), mask)
        drawing = ImageDraw.Draw(image)
        for outline in outlines:
            drawing.line(outline, fill=(*color, 255), width=max(2, image.width // 800), joint="curve")

    drawing = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, min(28, image.width // 60)))
    except OSError:
        font = ImageFont.load_default()
    lines = (f"Cortex tiles: {cortex_tiles:,}", f"Medulla tiles: {medullary_tiles:,}")
    line_height = max(drawing.textbbox((0, 0), line, font=font)[3] for line in lines) + 6
    text_width = max(drawing.textbbox((0, 0), line, font=font)[2] for line in lines)
    padding = 10
    box_width = text_width + 2 * padding
    box_height = 2 * line_height + 2 * padding
    left = max(0, image.width - box_width - padding)
    top = padding
    drawing.rounded_rectangle(
        (left, top, min(image.width, left + box_width), min(image.height, top + box_height)),
        radius=6, fill=(0, 0, 0, 210),
    )
    for index, line in enumerate(lines):
        drawing.text((left + padding, top + padding + index * line_height), line,
                     fill=(255, 255, 255, 255), font=font)

    output_dir = job_dir / "gt_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{sample}.png"
    temporary = destination.with_suffix(".png.tmp")
    try:
        image.convert("RGB").save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

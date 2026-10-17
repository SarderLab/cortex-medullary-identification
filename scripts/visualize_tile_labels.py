"""Draw TRIDENT tile boxes on a WSI thumbnail, with optional class colors.

Overlap can be specified in pixels or percent. A denser tile grid can be
subsampled to a wider stride when the requested grid lies on its coordinates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_MAG_FOLDER = re.compile(
    r"^(?P<mag>.+)_(?P<patch>\d+)px_(?P<overlap>\d+)px_overlap$"
)
_COLORS = {0: (70, 70, 70, 140), 1: (0, 102, 255, 220), 2: (255, 0, 0, 220)}


def _overlap_pixels(value: str, patch_size: int) -> int:
    if patch_size <= 0:
        raise ValueError("H5 patch_size must be positive")
    try:
        if value.endswith("%"):
            percentage = float(value[:-1])
            if not np.isfinite(percentage):
                raise ValueError
            overlap = round(patch_size * percentage / 100)
        else:
            overlap = int(value)
    except ValueError as error:
        raise ValueError("--overlap must be a pixel count (e.g. 256) or percent (e.g. 50%)") from error
    if not 0 <= overlap < patch_size:
        raise ValueError(f"--overlap must be between 0 and {patch_size - 1} pixels")
    return overlap


def _select_h5(input_h5: Path, requested: int, patch_size: int) -> tuple[Path, int, int]:
    match = _MAG_FOLDER.fullmatch(input_h5.parent.parent.name)
    if match is None or int(match.group("patch")) != patch_size:
        raise ValueError(
            "H5 must be under JOB_DIR/<magnification>_<patch>px_<overlap>px_overlap/"
            "features_<encoder>/ with a folder patch size matching its metadata"
        )
    job_dir = input_h5.parents[2]
    choices: list[tuple[int, int, Path]] = []
    for folder in job_dir.iterdir():
        folder_match = _MAG_FOLDER.fullmatch(folder.name)
        if (not folder.is_dir() or folder_match is None or
                folder_match.group("mag") != match.group("mag") or
                int(folder_match.group("patch")) != patch_size):
            continue
        source_overlap = int(folder_match.group("overlap"))
        source_stride = patch_size - source_overlap
        if source_stride <= 0:
            continue
        candidate = folder / input_h5.parent.name / input_h5.name
        if not candidate.is_file():
            continue
        # Every integer multiple of the source stride is a realizable sparser
        # grid. Pick the closest realizable overlap at or below the request.
        for factor in range(1, patch_size // source_stride + 1):
            displayed_overlap = patch_size - factor * source_stride
            if displayed_overlap <= requested:
                choices.append((displayed_overlap, -source_overlap, candidate))
    if not choices:
        raise ValueError(
            f"No available tile grid can display {requested}px overlap or a lower overlap "
            f"for {input_h5.name}; the requested stride must be an integer "
            "multiple of an available source stride"
        )
    displayed_overlap, negative_source_overlap, selected = max(choices)
    return selected, -negative_source_overlap, displayed_overlap


def _read_tile_data(path: Path, expected_overlap: int) -> tuple[np.ndarray, np.ndarray | None, int, tuple[int, int]]:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise ValueError(f"H5 needs a coords dataset: {path}")
        coords_dataset = handle["coords"]
        coords = np.asarray(coords_dataset[:])
        labels = np.asarray(handle["labels"][:]) if "labels" in handle else None
        attrs = coords_dataset.attrs
        tile_size_value = attrs.get("patch_size_level0", handle.attrs.get("patch_size_level0"))
        width_value = attrs.get("level0_width", handle.attrs.get("level0_width"))
        height_value = attrs.get("level0_height", handle.attrs.get("level0_height"))
        overlap_value = attrs.get("overlap")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if labels is not None and labels.shape != (len(coords),):
        raise ValueError("labels must have shape (N,)")
    if not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all():
        raise ValueError("coords must contain finite integer level-zero pixel origins")
    if labels is not None and not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("labels must contain only 0 (unlabeled), 1 (cortex), or 2 (medullary)")
    if tile_size_value is None or width_value is None or height_value is None:
        raise ValueError("H5 coords need patch_size_level0, level0_width, and level0_height attributes")
    tile_size = int(tile_size_value)
    dimensions = (int(width_value), int(height_value))
    if tile_size <= 0 or any(value <= 0 for value in dimensions):
        raise ValueError("Tile size and WSI dimensions must be positive")
    if overlap_value is not None and int(overlap_value) != expected_overlap:
        raise ValueError(f"H5 overlap attribute {overlap_value} disagrees with its folder name")
    return coords.astype(np.int64), (labels.astype(np.int8) if labels is not None else None), tile_size, dimensions


def _subsample_tiles(coords: np.ndarray, labels: np.ndarray | None, *,
                     patch_size: int, tile_size: int, source_overlap: int,
                     displayed_overlap: int) -> tuple[np.ndarray, np.ndarray | None]:
    if displayed_overlap == source_overlap or len(coords) == 0:
        return coords, labels
    source_stride = (patch_size - source_overlap) * tile_size / patch_size
    display_stride = (patch_size - displayed_overlap) * tile_size / patch_size
    source_pixels = round(source_stride)
    display_pixels = round(display_stride)
    if (not np.isclose(source_stride, source_pixels) or
            not np.isclose(display_stride, display_pixels) or
            source_pixels <= 0 or display_pixels % source_pixels):
        raise ValueError("Requested overlap cannot form an exact level-zero pixel stride")
    origin = coords.min(axis=0)
    if np.any((coords - origin) % source_pixels):
        raise ValueError("H5 tile origins are not on one regular source grid; cannot subsample exactly")
    keep = ((coords[:, 0] - origin[0]) % display_pixels == 0) & (
        (coords[:, 1] - origin[1]) % display_pixels == 0
    )
    return coords[keep], (labels[keep] if labels is not None else None)


def _draw_legend(image: Image.Image, counts: tuple[int, int, int],
                 overlap: int, patch_size: int, has_labels: bool) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, min(26, image.width // 65)))
    except OSError:
        font = ImageFont.load_default()
    if has_labels:
        lines = (f"Cortex: {counts[1]:,} tiles",
                 f"Medullary: {counts[2]:,} tiles",
                 f"Unlabeled: {counts[0]:,} tiles")
        colors = (_COLORS[1], _COLORS[2], _COLORS[0])
    else:
        lines = (f"Tiles: {counts[0]:,} (unlabeled)",)
        colors = (_COLORS[0],)
    lines += (f"Overlap: {overlap / patch_size:.0%} ({overlap} px)",)
    line_height = max(draw.textbbox((0, 0), line, font=font)[3] for line in lines) + 7
    width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 38
    padding = 10
    left = max(0, image.width - width - padding)
    draw.rounded_rectangle((left, padding, image.width - padding,
                            padding * 2 + line_height * len(lines)),
                           radius=6, fill=(0, 0, 0, 220))
    for index, line in enumerate(lines):
        y = padding * 2 + index * line_height
        if index < len(colors):
            draw.rectangle((left + 8, y + 3, left + 17, y + 12),
                           fill=colors[index])
        draw.text((left + 23, y), line, font=font, fill=(255, 255, 255, 255))


def visualize(input_h5: Path, overlap_arg: str, *, thumbnail: Path | None = None,
              output: Path | None = None, show: bool = False) -> tuple[Path, Path, int]:
    input_h5 = input_h5.expanduser().resolve()
    if not input_h5.is_file():
        raise FileNotFoundError(f"H5 file not found: {input_h5}")
    with h5py.File(input_h5, "r") as handle:
        if "coords" not in handle or "patch_size" not in handle["coords"].attrs:
            raise ValueError("Input H5 coords need a patch_size attribute")
        patch_size = int(handle["coords"].attrs["patch_size"])
    requested = _overlap_pixels(overlap_arg, patch_size)
    selected_h5, source_overlap, actual_overlap = _select_h5(input_h5, requested, patch_size)
    coords, labels, tile_size, dimensions = _read_tile_data(selected_h5, source_overlap)
    coords, labels = _subsample_tiles(
        coords, labels, patch_size=patch_size, tile_size=tile_size,
        source_overlap=source_overlap, displayed_overlap=actual_overlap,
    )

    job_dir = selected_h5.parents[2]
    thumbnail = (thumbnail.expanduser().resolve() if thumbnail else
                 job_dir / "thumbnails" / f"{selected_h5.stem}.jpg")
    if not thumbnail.is_file():
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail}; use --thumbnail PATH")
    with Image.open(thumbnail) as source:
        image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x = image.width / dimensions[0]
    scale_y = image.height / dimensions[1]
    line_width = max(1, image.width // 1200)
    for label in (0, 1, 2):
        selected_coords = coords if labels is None and label == 0 else (
            coords[labels == label] if labels is not None else coords[:0]
        )
        for x, y in selected_coords:
            rectangle = (round(x * scale_x), round(y * scale_y),
                         round((x + tile_size) * scale_x),
                         round((y + tile_size) * scale_y))
            draw.rectangle(rectangle, outline=_COLORS[label], width=line_width)
    image = Image.alpha_composite(image, overlay)
    counts = ((len(coords), 0, 0) if labels is None else
              tuple(int(np.count_nonzero(labels == value)) for value in (0, 1, 2)))
    _draw_legend(image, counts, actual_overlap, patch_size, labels is not None)

    output = (output.expanduser().resolve() if output else
              job_dir / "tile_label_visualization" /
              f"{selected_h5.stem}_overlap_{actual_overlap}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output)
    if show:
        image.show()
    return output, selected_h5, actual_overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="TRIDENT feature H5 for the sample")
    parser.add_argument("--overlap", required=True,
                        help="Desired overlap in pixels or percent, e.g. 0, 256, 50%%, or 75%%")
    parser.add_argument("--thumbnail", type=Path,
                        help="WSI thumbnail JPEG (default: JOB_DIR/thumbnails/SAMPLE.jpg)")
    parser.add_argument("--output", type=Path,
                        help="Output PNG path (default: JOB_DIR/tile_label_visualization/SAMPLE_overlap_N.png)")
    parser.add_argument("--show", action="store_true", help="Open the generated PNG in an image viewer")
    args = parser.parse_args()
    try:
        output, selected_h5, actual_overlap = visualize(
            args.h5, args.overlap, thumbnail=args.thumbnail,
            output=args.output, show=args.show,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Using H5: {selected_h5}")
    print(f"Source overlap: {_MAG_FOLDER.fullmatch(selected_h5.parent.parent.name).group('overlap')} px")
    print(f"Displayed overlap: {actual_overlap} px")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

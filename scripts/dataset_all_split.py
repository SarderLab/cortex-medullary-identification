"""Build a WSI workbook from slides that have matching TRIDENT feature H5 files."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


BLUE_ROOT = Path("/blue/pinaki.sarder/j.fermin/Cort_Med_Project")
ORANGE_ROOT = Path("/orange/pinaki.sarder/j.fermin/cort_med_wsi")
UF_CKD_DIR = Path(
    "/orange/pinaki.sarder/Davy_Jones_Locker/MMAI_OT2_Data/UF/"
    "Kidney_Biopsy_Images/Converted_TIFF_Images"
)
DEFAULT_WSI_DIRS = tuple(
    BLUE_ROOT / project / "Labelled_files"
    for project in ("KPMP_barisoni", "KPMP_hodgin", "KPMP_pathologist3")
) + tuple(
    ORANGE_ROOT / f"KPMP_batch{batch}" / "Labelled_files"
    for batch in range(2, 7)
) + (
    UF_CKD_DIR,
)
DEFAULT_H5_DIR = (
    ORANGE_ROOT / "trident_processed_final/KPMP/40x_512px_384px_overlap/features_uni_v1"
)
WSI_EXTENSIONS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".bif", ".svslide", ".isyntax", ".czi", ".dcm", ".qptif", ".qptiff",
    ".sld", ".kfb",
}
REQUIRED_COLUMNS = {
    "sample_name", "folder_path", "split", "num_cortex_tiles", "num_medullary_tiles",
}


def build_workbook(wsi_dirs: list[Path], h5_dir: Path, source: Path,
                   output: Path) -> tuple[int, int, int]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    h5_dir = h5_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source workbook not found: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not h5_dir.is_dir():
        raise NotADirectoryError(f"H5 feature directory not found: {h5_dir}")

    roots = [path.expanduser().resolve() for path in wsi_dirs]
    if not roots:
        raise ValueError("At least one WSI directory is required")
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(f"WSI directory not found: {root}")

    frame = pd.read_excel(source, dtype={"sample_name": str},
                          keep_default_na=False, engine="openpyxl")
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing workbook columns: {sorted(missing_columns)}")
    existing = {}
    for row in frame.to_dict("records"):
        stem = Path(row["sample_name"]).stem
        if not stem or stem in existing:
            raise ValueError(f"Empty or duplicate sample stem in source workbook: {stem!r}")
        existing[stem] = row

    h5_stems = {path.stem for path in h5_dir.glob("*.h5")}
    if not h5_stems:
        raise ValueError(f"No H5 files found in {h5_dir}")
    candidates: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for directory, subdirs, filenames in os.walk(root):
            subdirs.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix.lower() in WSI_EXTENSIONS and path.stem in h5_stems:
                    candidates[path.stem].append(path)
    if not candidates:
        raise ValueError("No WSIs with matching H5 files were found")

    rows = []
    duplicate_stems = 0
    for stem in sorted(candidates):
        paths = sorted(set(candidates[stem]))
        duplicate_stems += len(paths) > 1
        original = existing.get(stem)
        preferred = Path(original["folder_path"]) if original and original["folder_path"] else None
        path = preferred if preferred in paths else paths[0]
        if original:
            row = original.copy()
            row["folder_path"] = str(path)
        else:
            row = dict.fromkeys(frame.columns, "")
            row.update(sample_name=path.name, folder_path=str(path), split="holdout",
                       num_cortex_tiles=0.0, num_medullary_tiles=0.0)
        if path.is_relative_to(UF_CKD_DIR):
            row["split"] = "UF_CKD"
        rows.append(row)

    result = pd.DataFrame(rows, columns=frame.columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, index=False)
        sheet = writer.sheets["Sheet1"]
        for column in ("num_cortex_tiles", "num_medullary_tiles"):
            column_index = frame.columns.get_loc(column) + 1
            for cells in sheet.iter_cols(min_col=column_index, max_col=column_index,
                                         min_row=2):
                for cell in cells:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "0.0"
    retained = len(set(candidates) & set(existing))
    return retained, len(rows) - retained, duplicate_stems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-dirs", nargs="+", type=Path, default=DEFAULT_WSI_DIRS,
                        help="Directories to search recursively (default: KPMP labeled roots, batches 2–6, and MMAI OT2)")
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR,
                        help="TRIDENT features_uni_v1 directory containing matching .h5 files")
    parser.add_argument("--source", type=Path, default=Path("dataset_split.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("dataset_all_split.xlsx"))
    args = parser.parse_args()
    retained, added, duplicates = build_workbook(args.wsi_dirs, args.h5_dir,
                                                  args.source, args.output)
    print(f"Saved {args.output}: {retained} existing rows, {added} new rows; "
          f"{duplicates} slide names had multiple WSI paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

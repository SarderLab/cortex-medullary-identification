"""Add labeled H5 tile counts to dataset_split.xlsx without changing sample/WSI paths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from openpyxl import load_workbook

DEFAULT_REPORT = Path("/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed_final/KPMP/batch_labelling_report.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", type=Path, default=Path("dataset_split.xlsx"))
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = p.parse_args()
    counts = {}
    with args.report.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] == "processed":
                counts[row["sample"]] = (int(row["cortex_tiles"]), int(row["medullary_tiles"]))
    workbook = load_workbook(args.xlsx)
    sheet = workbook.active
    headers = {str(cell.value): cell.column for cell in sheet[1]}
    for name in ("sample_name", "folder_path", "split"):
        if name not in headers:
            raise ValueError(f"Missing workbook column: {name}")
    for name in ("num_cortex_tiles", "num_medullary_tiles"):
        if name not in headers:
            column = sheet.max_column + 1
            sheet.cell(1, column, name)
            headers[name] = column
    matched = 0
    for cells in sheet.iter_rows(min_row=2):
        sample = str(cells[headers["sample_name"] - 1].value)
        stem = Path(sample).stem
        split_cell = cells[headers["split"] - 1]
        if split_cell.value not in ("train", "val", "test", "NA"):
            split_cell.value = "holdout"
        pair = counts.get(stem)
        if pair is not None:
            matched += 1
        for name, value in zip(("num_cortex_tiles", "num_medullary_tiles"), pair or (None, None)):
            sheet.cell(cells[0].row, headers[name], value)
    workbook.save(args.xlsx)
    print(f"Updated {args.xlsx}: {matched}/{sheet.max_row - 1} samples have labeled H5 tile counts")


if __name__ == "__main__":
    main()

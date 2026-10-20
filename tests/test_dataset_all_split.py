"""Check the all-slide workbook against nested WSI and H5 fixtures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

from scripts.dataset_all_split import build_workbook


class AllSplitWorkbookTests(unittest.TestCase):
    def test_preserves_existing_rows_and_adds_only_h5_matched_slides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            h5_dir = root / "features_uni_v1"
            for folder in (first, second, h5_dir):
                folder.mkdir()
            old_path = first / "nested" / "old.svs"
            old_path.parent.mkdir()
            old_path.touch()
            duplicate = second / "old.svs"
            duplicate.touch()
            new_path = second / "nested" / "new.tiff"
            new_path.parent.mkdir()
            new_path.touch()
            (second / "mask.tif").touch()
            for stem in ("old", "new"):
                (h5_dir / f"{stem}.h5").touch()

            source = root / "dataset_split.xlsx"
            output = root / "dataset_all_split.xlsx"
            pd.DataFrame([
                {"sample_name": "old.svs", "folder_path": str(old_path), "split": "val",
                 "num_cortex_tiles": 12.0, "num_medullary_tiles": 3.0,
                 "extra": "keep"},
                {"sample_name": "absent.svs", "folder_path": "", "split": "train",
                 "num_cortex_tiles": 4.0, "num_medullary_tiles": 2.0,
                 "extra": "omit"},
            ]).to_excel(source, index=False)

            self.assertEqual(build_workbook([first, second], h5_dir, source, output),
                             (1, 1, 1))
            result = pd.read_excel(output, keep_default_na=False)
            self.assertEqual(result.columns.tolist(), [
                "sample_name", "folder_path", "split", "num_cortex_tiles",
                "num_medullary_tiles", "extra",
            ])
            self.assertEqual(set(result.sample_name), {"old.svs", "new.tiff"})
            old = result.set_index("sample_name").loc["old.svs"]
            new = result.set_index("sample_name").loc["new.tiff"]
            self.assertEqual((old.folder_path, old.split, old.num_cortex_tiles,
                              old.num_medullary_tiles, old.extra),
                             (str(old_path), "val", 12.0, 3.0, "keep"))
            self.assertEqual((new.folder_path, new.split, new.num_cortex_tiles,
                              new.num_medullary_tiles),
                             (str(new_path), "holdout", 0.0, 0.0))
            sheet = load_workbook(output, read_only=False).active
            self.assertEqual(sheet["D2"].number_format, "0.0")
            self.assertEqual(sheet["E2"].number_format, "0.0")
            with self.assertRaises(FileExistsError):
                build_workbook([first, second], h5_dir, source, output)

    def test_uf_slides_use_uf_ckd_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uf_dir = root / "Converted_TIFF_Images"
            nested = uf_dir / "nested"
            h5_dir = root / "features_uni_v1"
            nested.mkdir(parents=True)
            h5_dir.mkdir()
            old_path = nested / "old.tif"
            new_path = nested / "new.tiff"
            old_path.touch()
            new_path.touch()
            for stem in ("old", "new"):
                (h5_dir / f"{stem}.h5").touch()
            source = root / "dataset_split.xlsx"
            output = root / "dataset_all_split.xlsx"
            pd.DataFrame([{
                "sample_name": "old.tif", "folder_path": str(old_path), "split": "val",
                "num_cortex_tiles": 7.0, "num_medullary_tiles": 2.0,
            }]).to_excel(source, index=False)

            with patch("scripts.dataset_all_split.UF_CKD_DIR", uf_dir):
                self.assertEqual(build_workbook([uf_dir], h5_dir, source, output),
                                 (1, 1, 0))
            rows = pd.read_excel(output).set_index("sample_name")
            self.assertEqual(rows.loc["old.tif", "split"], "UF_CKD")
            self.assertEqual(rows.loc["old.tif", "num_cortex_tiles"], 7.0)
            self.assertEqual(rows.loc["old.tif", "num_medullary_tiles"], 2.0)
            self.assertEqual(rows.loc["new.tiff", "split"], "UF_CKD")
            self.assertEqual(rows.loc["new.tiff", "num_cortex_tiles"], 0.0)
            self.assertEqual(rows.loc["new.tiff", "num_medullary_tiles"], 0.0)


if __name__ == "__main__":
    unittest.main()

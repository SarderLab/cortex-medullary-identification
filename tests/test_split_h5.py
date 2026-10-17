"""Check that the generated split workbook is usable by the H5 SVC loader."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np

from svc.h5_svm import workbook_rows
from svc.split_h5 import make_splits


class SplitWorkbookTests(unittest.TestCase):
    def test_xlsx_contains_three_splits_and_matches_h5_files(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            features_dir = job / "40x_512px_256px_overlap" / "features_uni_v1"
            features_dir.mkdir(parents=True)
            for name in ("slide_a", "slide_b", "slide_c"):
                with h5py.File(features_dir / f"{name}.h5", "w") as handle:
                    handle.create_dataset("features", data=np.ones((2, 4), dtype=np.float32))
                    handle.create_dataset("coords", data=np.array([[0, 0], [256, 0]]))
                    handle.create_dataset("labels", data=np.array([1, 2]))
            output = job / "dataset_split.xlsx"
            make_splits(job, output, features_dir=features_dir)
            self.assertTrue(zipfile.is_zipfile(output))
            for split in ("train", "val", "test"):
                rows = workbook_rows(output, features_dir.parent, split)
                self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()

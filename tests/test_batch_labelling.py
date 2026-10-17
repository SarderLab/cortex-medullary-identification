"""End-to-end check for optional batch GT thumbnail output."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
from shapely.geometry import box, mapping

from svc.batch_labelling import main


class BatchLabellingVisualizationTests(unittest.TestCase):
    def test_labels_and_gt_overlay_use_the_same_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations" / "sample"
            annotations.mkdir(parents=True)
            for name, geometry in (
                ("cortical_gt.json", box(0, 0, 50, 100)),
                ("medullary_gt.json", box(50, 0, 100, 100)),
            ):
                (annotations / name).write_text(json.dumps(mapping(geometry)), encoding="utf-8")

            job = root / "job"
            feature_dir = job / "40x_50px_0px_overlap" / "features_uni_v1"
            feature_dir.mkdir(parents=True)
            with h5py.File(feature_dir / "sample.h5", "w") as handle:
                handle.create_dataset("features", data=np.ones((2, 3), dtype=np.float32))
                coords = handle.create_dataset("coords", data=np.array([[0, 0], [50, 0]]))
                coords.attrs["patch_size_level0"] = 50
                coords.attrs["level0_width"] = 100
                coords.attrs["level0_height"] = 100
            (job / "thumbnails").mkdir()
            Image.new("RGB", (200, 200), (160, 160, 160)).save(job / "thumbnails" / "sample.jpg")

            with patch.object(sys, "argv", [
                "batch_labelling", "--annotation-dir", str(root / "annotations"),
                "--job-dir", str(job), "--save-gt-visualization",
            ]):
                self.assertEqual(main(), 0)

            with h5py.File(feature_dir / "sample.h5", "r") as handle:
                self.assertEqual(handle["labels"][:].tolist(), [1, 2])
            with (job / "batch_labelling_report.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual((row["cortex_tiles"], row["medullary_tiles"]), ("1", "1"))
            visualization = job / "gt_visualization" / "sample.png"
            self.assertEqual(row["visualization_path"], str(visualization))
            with Image.open(visualization) as image:
                self.assertEqual(image.size, (200, 200))
                self.assertGreater(image.getpixel((40, 120))[2], image.getpixel((40, 120))[0])
                self.assertGreater(image.getpixel((160, 120))[0], image.getpixel((160, 120))[2])


if __name__ == "__main__":
    unittest.main()

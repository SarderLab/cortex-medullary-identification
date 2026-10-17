"""Verify tile box colors and requested-overlap fallback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.visualize_tile_labels import visualize


def _feature_h5(job: Path, sample: str, overlap: int, *, patch_size: int = 512,
                coords: np.ndarray | None = None, labeled: bool = True) -> Path:
    directory = job / f"40x_{patch_size}px_{overlap}px_overlap" / "features_uni_v1"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample}.h5"
    with h5py.File(path, "w") as handle:
        if coords is None:
            coords = np.array([[0, 0], [patch_size, 0]])
        coordinate_data = handle.create_dataset("coords", data=coords)
        coordinate_data.attrs["patch_size"] = patch_size
        coordinate_data.attrs["patch_size_level0"] = patch_size
        coordinate_data.attrs["level0_width"] = patch_size * 2
        coordinate_data.attrs["level0_height"] = patch_size * 2
        coordinate_data.attrs["overlap"] = overlap
        if labeled:
            handle.create_dataset("labels", data=np.resize(np.array([1, 2], dtype=np.int8), len(coords)))
    return path


class TileLabelVisualizationTests(unittest.TestCase):
    def test_75_percent_falls_back_to_50_percent_and_draws_class_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            source = _feature_h5(job, "sample", 0)
            selected = _feature_h5(job, "sample", 256)
            (job / "thumbnails").mkdir()
            Image.new("RGB", (500, 500), "white").save(job / "thumbnails" / "sample.jpg")

            output, actual_h5, overlap = visualize(source, "75%")
            self.assertEqual((actual_h5, overlap), (selected, 256))
            self.assertEqual(output, job / "tile_label_visualization" / "sample_overlap_256.png")
            with Image.open(output) as image:
                self.assertGreater(image.getpixel((125, 250))[2], image.getpixel((125, 250))[0])
                self.assertGreater(image.getpixel((375, 250))[0], image.getpixel((375, 250))[2])

    def test_50_percent_falls_back_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            source = _feature_h5(job, "sample", 0)
            (job / "thumbnails").mkdir()
            Image.new("RGB", (200, 200), "white").save(job / "thumbnails" / "sample.jpg")
            _, selected, overlap = visualize(source, "256")
            self.assertEqual((selected, overlap), (source, 0))

    def test_unlabeled_50_percent_grid_can_be_displayed_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            grid = np.array([(x, y) for y in (0, 128, 256) for x in (0, 128, 256)])
            source = _feature_h5(job, "sample", 128, patch_size=256,
                                 coords=grid, labeled=False)
            (job / "thumbnails").mkdir()
            Image.new("RGB", (500, 500), "white").save(job / "thumbnails" / "sample.jpg")

            output, selected, overlap = visualize(source, "0")
            self.assertEqual((selected, overlap), (source, 0))
            with Image.open(output) as image:
                # The 128-pixel origin (125 thumbnail pixels) was removed.
                self.assertEqual(image.getpixel((125, 125)), (255, 255, 255))
                # The 256-pixel boundary of the retained tile is gray.
                gray = image.getpixel((125, 250))
                self.assertLess(gray[0], 255)
                self.assertEqual(gray[0], gray[1])


if __name__ == "__main__":
    unittest.main()

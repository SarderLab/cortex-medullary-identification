"""CPU-only tests for WSI patch planning, polygon rasterization, and native HDF5 IO."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from shapely.geometry import box, mapping

from svc.single_wsi import _save_metrics_txt, _validate_physical_tile_size, load_feature_h5
from svc.tile_annotations import label_tiles, load_annotation_geometry, write_tile_annotation_csv
from svc.wsi_patch_plan import patch_plan


class PatchPlanTests(unittest.TestCase):
    def test_physical_field_and_overlap_for_40x_and_20x(self):
        at_40 = patch_plan(
            {"openslide.objective-power": "40", "openslide.mpp-x": "0.25"}, 75,
        )
        self.assertEqual((at_40["magnification"], at_40["patch_size"], at_40["overlap_px"]), (40, 512, 384))
        self.assertEqual(at_40["encoder_img_size"], 512)

        at_20 = patch_plan(
            {"openslide.objective-power": "20", "openslide.mpp-x": "0.5"}, 50,
        )
        self.assertEqual((at_20["magnification"], at_20["patch_size"], at_20["overlap_px"]), (20, 256, 128))
        self.assertEqual(at_20["encoder_img_size"], 512)

    def test_patch_size_uses_mpp_and_rejects_unknown_overlap(self):
        plan = patch_plan({"openslide.objective-power": "40", "openslide.mpp-x": "0.30"}, 0)
        self.assertEqual(plan["patch_size"], 427)
        with self.assertRaises(ValueError):
            patch_plan({"openslide.objective-power": "40", "openslide.mpp-x": "0.25"}, 25)


class AnnotationTests(unittest.TestCase):
    def test_geojson_rasterization_and_medulla_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cortical_gt.json").write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 256, 256))}],
            }))
            (root / "medullary_gt.json").write_text(json.dumps({
                "document": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 128, 256))}],
                }
            }))
            cortex = load_annotation_geometry(root / "cortical_gt.json")
            medullary = load_annotation_geometry(root / "medullary_gt.json")
            coords = np.asarray([[0, 0], [256, 0], [0, 256]], dtype=np.int64)
            labels, cortex_overlap, medullary_overlap = label_tiles(coords, 256, cortex, medullary)
            self.assertEqual(labels.tolist(), [2, 0, 0])
            self.assertEqual(cortex_overlap.tolist(), [True, False, False])
            self.assertEqual(medullary_overlap.tolist(), [True, False, False])

    def test_tile_csv_preserves_row_order_and_raw_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tile_annotations" / "slide.csv"
            coords = np.asarray([[10, 20], [30, 40]], dtype=np.int64)
            labels = np.asarray([1, 2], dtype=np.int8)
            overlap = np.asarray([True, False])
            write_tile_annotation_csv(path, coords, 256, labels, overlap, ~overlap)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["tile_index"] for row in rows], ["0", "1"])
            self.assertEqual([row["label"] for row in rows], ["1", "2"])
            self.assertEqual([row["label_name"] for row in rows], ["cortex", "medullary"])


class Hdf5AndMetricsTests(unittest.TestCase):
    def test_native_hdf5_is_read_without_pt_conversion_and_fov_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "sample.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("features", data=np.ones((2, 4), dtype=np.float32))
                coordinates = handle.create_dataset("coords", data=np.asarray([[0, 0], [256, 0]], dtype=np.int64))
                coordinates.attrs["patch_size_level0"] = 512
                handle.attrs["encoder"] = "uni_v1"
            before = h5_path.read_bytes()
            features, coords, tile_size, _, _, encoder = load_feature_h5(h5_path)
            self.assertEqual(features.shape, (2, 4))
            self.assertEqual(coords.tolist(), [[0, 0], [256, 0]])
            self.assertEqual((tile_size, encoder), (512, "uni_v1"))
            self.assertEqual(h5_path.read_bytes(), before)
            self.assertEqual(list(root.glob("*.pt")), [])
            self.assertEqual(_validate_physical_tile_size(512, mpp_x=0.25, mpp_y=0.25,
                             expected_microns=128.0, maximum_error_percent=1.0), (128.0, 128.0))
            with self.assertRaises(ValueError):
                _validate_physical_tile_size(512, mpp_x=0.5, mpp_y=0.5,
                                             expected_microns=128.0, maximum_error_percent=10.0)

    def test_metrics_text_includes_auc_per_class_and_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = _save_metrics_txt(
                Path(directory), sample_name="sample", wsi_path=Path("sample.svs"),
                annotation_dir=Path("annotations/sample"), feature_h5=Path("features.h5"),
                y_true=np.asarray([0, 0, 1, 1]), y_pred=np.asarray([0, 1, 1, 1]),
                scores=np.asarray([-1.0, 0.2, 0.7, 1.0]), threshold=0.0,
            )
            content = output.read_text(encoding="utf-8")
            self.assertRegex(output.name, r"^\d{4}-\d{2}-\d{2}\.txt$")
            for text in ("Overall accuracy:", "Balanced accuracy:", "cortex: accuracy=", "AUC=",
                         "sensitivity=", "specificity=", "Confusion matrix"):
                self.assertIn(text, content)


if __name__ == "__main__":
    unittest.main()

"""Check slide-level splits and direct loading of labeled Trident features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from svc.data import load_labeled_h5_dataset
from svc.manifest import load_h5_split_samples
from svc.split_h5 import make_splits
from svc import train as train_module


def write_feature(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.arange(12, dtype=np.float32).reshape(4, 3))
        handle.create_dataset("coords", data=np.array([[0, 0], [1, 0], [2, 0], [3, 0]]))
        handle.create_dataset("labels", data=np.array([0, 1, 2, 1], dtype=np.int8))


class LabeledH5TrainingTests(unittest.TestCase):
    def test_split_by_wsi_and_load_labeled_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                write_feature(root / "output" / "40x_512px_256px_overlap" /
                              "features_uni_v1" / f"sample_{index}.h5")
            manifest = root / "output" / "splits.csv"
            rows = make_splits(root / "output", manifest)
            self.assertEqual({split: sum(row["split"] == split for row in rows)
                              for split in ("train", "val", "test")},
                             {"train": 7, "val": 2, "test": 1})
            self.assertEqual(rows, make_splits(root / "output", manifest))

            seen: set[str] = set()
            for split, expected_count in (("train", 7), ("val", 2), ("test", 1)):
                frame = load_h5_split_samples(manifest, split)
                self.assertEqual(len(frame), expected_count)
                self.assertFalse(seen.intersection(frame["sample_name"]))
                seen.update(frame["sample_name"])
                dataset = load_labeled_h5_dataset(frame)
                self.assertEqual(dataset.features.shape, (expected_count * 3, 3))
                self.assertEqual(set(dataset.labels.tolist()), {0, 1})
                self.assertTrue(np.all(dataset.coordinates[:, 0] >= 1))

    def test_duplicate_feature_set_needs_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for encoder in ("uni_v1", "uni_v2"):
                write_feature(root / "output" / "40x_512px_256px_overlap" /
                              f"features_{encoder}" / "sample.h5")
            with self.assertRaisesRegex(ValueError, "Multiple labeled feature files"):
                make_splits(root / "output", root / "splits.csv")

    def test_training_reads_h5_and_reports_all_three_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                write_feature(root / "output" / "40x_512px_256px_overlap" /
                              "features_uni_v1" / f"sample_{index}.h5")
            manifest = root / "output" / "splits.csv"
            make_splits(root / "output", manifest)
            fitted_rows: list[int] = []

            def fake_fit(features, labels, **kwargs):
                fitted_rows.append(len(labels))
                return {"gpu": {"name": "test"}}

            def fake_predict(bundle, features, **kwargs):
                score = np.where(features.numpy()[:, 0] == 6, 1.0, -1.0)
                return (score >= 0).astype(np.int32), score.astype(np.float32)

            def fake_save(bundle, path, metadata):
                path.write_bytes(b"test model")
                return path

            def fake_confusion(metrics, path):
                path.write_bytes(b"test chart")

            model_dir = root / "models"
            with patch.object(train_module, "train_gpu_pipeline", side_effect=fake_fit), \
                 patch.object(train_module, "predict_gpu", side_effect=fake_predict), \
                 patch.object(train_module, "save_bundle", side_effect=fake_save), \
                 patch.object(train_module, "save_confusion_matrix", side_effect=fake_confusion):
                self.assertEqual(train_module.main(["--manifest", str(manifest),
                                                    "--output-dir", str(model_dir),
                                                    "--n-components", "2"]), 0)
            self.assertEqual(fitted_rows, [21])
            for name in ("svc_pipeline.pkl", "train_metrics.json", "val_metrics.json",
                         "test_metrics.json", "test_confusion_matrix.png",
                         "test_tile_predictions.csv"):
                self.assertTrue((model_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

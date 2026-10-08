import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anomaly.data import (contained, load_image, make_manifest, perturb,
                          select_training, training_views, validate_manifest)
from anomaly.metrics import auroc, classification_metrics
from anomaly.model import greedy_coreset, nearest_distances
from anomaly.run import validate_config


class PipelineTests(unittest.TestCase):
    def test_auroc_ties_against_pairwise_definition(self):
        labels = np.array([0, 1, 1, 0, 0, 1])
        scores = np.array([0.3, 0.5, 0.5, 0.5, 0.9, 0.8])
        positive, negative = scores[labels == 1], scores[labels == 0]
        expected = np.mean((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative))
        self.assertAlmostEqual(auroc(labels, scores), expected)
        self.assertEqual(auroc([0, 1], [1, 1]), 0.5)
        self.assertIsNone(auroc([0, 0], [0, 1]))

    def test_fixed_threshold_strict_greater(self):
        values = classification_metrics([0, 0, 1, 1], [0.5, 0.6, 0.5, 0.7], 0.5)
        self.assertEqual(values["false_positives"], 1)
        self.assertEqual(values["true_positives"], 1)
        self.assertEqual(values["recall"], 0.5)

    def test_chunked_nearest_against_brute_force(self):
        torch.manual_seed(9)
        query, bank = torch.randn(13, 4), torch.randn(17, 4)
        expected = ((query[:, None] - bank[None]) ** 2).sum(-1).sqrt().min(1).values
        torch.testing.assert_close(nearest_distances(query, bank, 3, 5), expected)

    def test_coreset_budget_and_unique_indices_with_identical_features(self):
        features = torch.zeros(12, 6)
        bank, indices = greedy_coreset(features, 7, 42, projection_dim=4)
        self.assertEqual(bank.shape, (7, 6))
        self.assertEqual(len(indices.unique()), 7)
        _, repeated = greedy_coreset(features, 7, 42, projection_dim=4)
        torch.testing.assert_close(indices, repeated)
        with self.assertRaises(ValueError):
            greedy_coreset(features, 13, 42)

    def test_perturbations_and_view_pairing(self):
        image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 11
        torch.testing.assert_close(perturb(image, "brightness", 1), image)
        torch.testing.assert_close(perturb(image, "contrast", 1), image)
        mean = image.mean((-2, -1), keepdim=True)
        torch.testing.assert_close(perturb(image, "contrast", 0.7), (image - mean) * 0.7 + mean)
        row = {"path": "bottle/train/good/001.png"}
        b = list(training_views(image, row, 42, 3))
        c = list(training_views(image, row, 42, 1))
        self.assertEqual(b[1][1], c[1][1])
        torch.testing.assert_close(b[1][0], c[1][0])

    def test_manifest_disjoint_reproducible_nested_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, defect, count, offset in [("train", "good", 40, 0), ("test", "good", 10, 50), ("test", "scratch", 10, 80)]:
                folder = root / "bottle" / split / defect
                folder.mkdir(parents=True)
                for i in range(count):
                    Image.fromarray(np.full((8, 8, 3), i + offset, dtype=np.uint8)).save(folder / f"{i:03}.png")
                    if defect != "good":
                        masks = root / "bottle/ground_truth" / defect
                        masks.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(masks / f"{i:03}_mask.png")
            manifest = make_manifest(root, ["bottle"])
            validate_manifest(manifest)
            self.assertEqual(manifest, make_manifest(root, ["bottle"]))
            short = select_training(manifest["rows"], "bottle", 42, 5)
            long = select_training(manifest["rows"], "bottle", 42, 25)
            self.assertEqual(short, long[:5])
            self.assertEqual(len(select_training(manifest["rows"], "bottle", 42, "all")), 32)
            self.assertEqual(load_image(root, short[0], 16).shape, (3, 16, 16))
            (root / short[0]["path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_image(root, short[0], 16)
            with self.assertRaises(ValueError):
                contained(root, "../escape.png")

    def test_final_test_rejected_before_io(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/bottle_pilot.json"
        config = json.loads(config_path.read_text())
        validate_config(config)
        config["evaluation_split"] = "final_test"
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

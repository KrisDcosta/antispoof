import unittest

import numpy as np

from src.evaluate import compute_eer, compute_metrics


class EvaluateTests(unittest.TestCase):
    def test_compute_eer_separates_ordered_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        eer, threshold = compute_eer(y_true, y_scores)

        self.assertAlmostEqual(eer, 0.0)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)

    def test_compute_metrics_uses_eer_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_metrics(y_true, y_scores)

        self.assertAlmostEqual(metrics["eer"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"].shape, (2, 2))

    def test_eer_requires_two_classes(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([1, 1]), np.array([0.8, 0.9]))


if __name__ == "__main__":
    unittest.main()

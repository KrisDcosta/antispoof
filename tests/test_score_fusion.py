import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_score_fusion import (
    ScoreSource,
    add_normalized_scores,
    align_scores,
    fit_dev_normalization,
    parse_alpha_grid,
    run_fusion,
    select_weighted_alpha,
)


def write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "path", "label", "label_name", "system_id", "score"])
        writer.writeheader()
        writer.writerows(rows)


class ScoreFusionTests(unittest.TestCase):
    def test_align_scores_requires_consistent_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows_a = [{
                "file_id": "x",
                "path": "x.flac",
                "label": 1,
                "label_name": "bonafide",
                "system_id": "-",
                "score": 0.9,
            }]
            rows_b = [{**rows_a[0], "label": 0, "label_name": "spoof"}]
            write_scores(tmp_path / "a_dev.csv", rows_a)
            write_scores(tmp_path / "a_eval.csv", rows_a)
            write_scores(tmp_path / "b_dev.csv", rows_b)
            write_scores(tmp_path / "b_eval.csv", rows_b)

            sources = [
                ScoreSource("a", tmp_path / "a_dev.csv", tmp_path / "a_eval.csv"),
                ScoreSource("b", tmp_path / "b_dev.csv", tmp_path / "b_eval.csv"),
            ]

            with self.assertRaises(ValueError):
                align_scores(sources, "dev")

    def test_normalization_uses_dev_statistics_for_eval(self):
        dev_rows = [{"a_score": 1.0}, {"a_score": 3.0}]
        eval_rows = [{"a_score": 2.0}]

        stats = fit_dev_normalization(dev_rows, ["a"])
        normalized = add_normalized_scores(eval_rows, ["a"], stats)

        self.assertAlmostEqual(stats["a"].mean, 2.0)
        self.assertAlmostEqual(stats["a"].std, 1.0)
        self.assertAlmostEqual(normalized[0]["z_a"], 0.0)

    def test_weighted_alpha_is_selected_on_dev_eer(self):
        rows = [
            {"label": 0, "z_lcnn": 2.0, "z_wavlm": 0.0},
            {"label": 0, "z_lcnn": 2.0, "z_wavlm": 0.2},
            {"label": 1, "z_lcnn": 0.0, "z_wavlm": 0.8},
            {"label": 1, "z_lcnn": 0.0, "z_wavlm": 1.0},
        ]

        alpha, candidates = select_weighted_alpha(rows, "lcnn", "wavlm", [0.0, 0.5, 1.0])

        self.assertEqual(alpha, 0.0)
        self.assertEqual(len(candidates), 3)

    def test_run_fusion_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_rows = [
                {"file_id": "b1", "path": "b1.flac", "label": 1, "label_name": "bonafide", "system_id": "-", "score": 0.9},
                {"file_id": "b2", "path": "b2.flac", "label": 1, "label_name": "bonafide", "system_id": "-", "score": 0.8},
                {"file_id": "s1", "path": "s1.flac", "label": 0, "label_name": "spoof", "system_id": "A17", "score": 0.2},
                {"file_id": "s2", "path": "s2.flac", "label": 0, "label_name": "spoof", "system_id": "A18", "score": 0.1},
            ]
            alt_rows = [
                {**source_rows[0], "score": 0.7},
                {**source_rows[1], "score": 0.6},
                {**source_rows[2], "score": 0.3},
                {**source_rows[3], "score": 0.2},
            ]
            write_scores(tmp_path / "a_dev.csv", source_rows)
            write_scores(tmp_path / "a_eval.csv", source_rows)
            write_scores(tmp_path / "b_dev.csv", alt_rows)
            write_scores(tmp_path / "b_eval.csv", alt_rows)

            metrics = run_fusion(
                "unit_fusion",
                tmp_path / "fusion",
                [
                    ScoreSource("lcnn", tmp_path / "a_dev.csv", tmp_path / "a_eval.csv"),
                    ScoreSource("wavlm", tmp_path / "b_dev.csv", tmp_path / "b_eval.csv"),
                ],
                parse_alpha_grid("0,0.5,1"),
            )

            run_dir = tmp_path / "fusion" / "unit_fusion"
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "scores" / "eval_scores.csv").exists())
            self.assertTrue((tmp_path / "fusion" / "metrics" / "unit_fusion_summary.json").exists())
            self.assertIn(metrics["selected_method"], {"mean", "weighted_mean", "logistic_regression"})


if __name__ == "__main__":
    unittest.main()

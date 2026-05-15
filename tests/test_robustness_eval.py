import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.run_robustness_eval import (
    Corruption,
    FusionStats,
    apply_corruption,
    apply_frozen_fusion_score,
    load_fusion_stats,
    validate_config,
    write_csv,
)


class RobustnessEvalTests(unittest.TestCase):
    def test_gain_corruption_changes_amplitude(self):
        waveform = torch.tensor([0.5, -0.25], dtype=torch.float32)
        out = apply_corruption(waveform, Corruption("gain_p6db", "gain", {"db": 6}))
        self.assertTrue(torch.allclose(out, waveform * (10 ** (6 / 20)), atol=1e-6))

    def test_clipping_clamps_values(self):
        waveform = torch.tensor([-1.0, -0.2, 0.3, 1.0], dtype=torch.float32)
        out = apply_corruption(waveform, Corruption("clip", "clipping", {"threshold": 0.25}))
        self.assertLessEqual(float(out.max()), 0.25)
        self.assertGreaterEqual(float(out.min()), -0.25)

    def test_resampling_returns_original_shape(self):
        waveform = torch.linspace(-1.0, 1.0, 1600)
        out = apply_corruption(
            waveform,
            Corruption("resample_8k", "resample", {"target_sample_rate": 8000}),
            sample_rate=16000,
        )
        self.assertEqual(out.numel(), waveform.numel())

    def test_noise_is_deterministic_with_seed(self):
        waveform = torch.ones(512)
        corruption = Corruption("noise_10db", "noise", {"snr_db": 10})
        first = apply_corruption(waveform, corruption, seed=7, file_id="LA_E_1")
        second = apply_corruption(waveform, corruption, seed=7, file_id="LA_E_1")
        self.assertTrue(torch.equal(first, second))

    def test_fusion_uses_stored_stats(self):
        stats = FusionStats(lcnn_mean=10.0, lcnn_std=2.0, wavlm_mean=1.0, wavlm_std=4.0, alpha=0.7)
        score = apply_frozen_fusion_score(12.0, 5.0, stats)
        self.assertAlmostEqual(score, 0.7 * 1.0 + 0.3 * 1.0)

    def test_load_fusion_stats_falls_back_to_full_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            run_dir = root / "unit_fusion"
            metrics_dir.mkdir()
            run_dir.mkdir()
            (metrics_dir / "unit_fusion_summary.json").write_text(json.dumps({
                "methods": {"weighted_mean": {"rule": {"alpha": 0.7}}}
            }))
            (run_dir / "metrics.json").write_text(json.dumps({
                "score_normalization": {
                    "lcnn": {"mean": 1.0, "std": 2.0},
                    "wavlm": {"mean": 3.0, "std": 4.0}
                }
            }))
            stats = load_fusion_stats(metrics_dir / "unit_fusion_summary.json")
            self.assertEqual(stats.lcnn_mean, 1.0)
            self.assertEqual(stats.wavlm_std, 4.0)

    def test_output_artifact_writing_on_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv(path, [{"condition": "clean", "model": "lcnn", "eer": 0.1}])
            self.assertIn("condition,model,eer", path.read_text().splitlines()[0])

    def test_config_validation_catches_missing_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/LA/ASVspoof2019_LA_eval/flac").mkdir(parents=True)
            proto = root / "data/LA/ASVspoof2019_LA_cm_protocols"
            proto.mkdir(parents=True)
            (proto / "ASVspoof2019.LA.cm.eval.trl.txt").write_text("")
            cfg = root / "config.json"
            summary = root / "fusion_summary.json"
            cfg.write_text("{}")
            summary.write_text("{}")
            config = {
                "data_root": str(root / "data/LA"),
                "split": "eval",
                "systems": {
                    "lcnn": {"config_path": str(cfg), "checkpoint_path": str(root / "missing_lcnn.pt")},
                    "wavlm": {"config_path": str(cfg), "checkpoint_path": str(root / "missing_wavlm.pt")},
                    "fusion": {"summary_path": str(summary)},
                },
                "corruptions": [{"name": "clean", "type": "identity"}],
            }
            with self.assertRaises(FileNotFoundError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()

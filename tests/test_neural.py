import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.dataset import Sample
from src.neural.dataset import ASVspoofSpectrogramDataset, class_balanced_limit
from src.neural.models import build_model
from src.neural.transforms import LogMelTransform, crop_or_pad


class NeuralTests(unittest.TestCase):
    def test_crop_or_pad_returns_fixed_length(self):
        self.assertEqual(crop_or_pad(torch.ones(4), 8).shape[0], 8)
        self.assertEqual(crop_or_pad(torch.ones(12), 8).shape[0], 8)

    def test_logmel_transform_shape(self):
        transform = LogMelTransform(sample_rate=16_000, n_mels=64)
        x = transform(torch.zeros(16_000))

        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(x.shape[1], 64)
        self.assertTrue(torch.isfinite(x).all())

    def test_dataset_sample_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.wav"
            sf.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
            sample = Sample("sample", str(audio_path), 1, "-")
            dataset = ASVspoofSpectrogramDataset(
                [sample],
                LogMelTransform(sample_rate=16_000, n_mels=64),
                sample_rate=16_000,
                clip_seconds=1.0,
            )

            item = dataset[0]

            self.assertEqual(item["x"].shape[0], 1)
            self.assertEqual(item["x"].shape[1], 64)
            self.assertEqual(float(item["label"]), 1.0)
            self.assertEqual(item["file_id"], "sample")

    def test_model_forward_shape(self):
        model = build_model("lcnn", dropout=0.1)
        x = torch.randn(2, 1, 64, 401)

        y = model(x)

        self.assertEqual(y.shape, (2,))

    def test_class_balanced_limit(self):
        samples = [
            Sample(f"b{i}", f"b{i}.wav", 1, "-") for i in range(10)
        ] + [
            Sample(f"s{i}", f"s{i}.wav", 0, "A01") for i in range(20)
        ]

        selected = class_balanced_limit(samples, 8, seed=0)

        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(s.label for s in selected), 4)


if __name__ == "__main__":
    unittest.main()

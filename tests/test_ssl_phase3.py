import tempfile
import unittest
from pathlib import Path

import torch

from scripts.check_phase3_ssl_ready import validate_cache_payload, validate_config
from src.neural.ssl_dataset import SSLEmbeddingDataset, TrainMeanStdNormalizer, class_weights_from_labels
from src.neural.ssl_embeddings import ensure_cache_writable, pooled_mean_std, save_cache
from src.neural.ssl_models import SSLPooledMLP


class SSLPhase3Tests(unittest.TestCase):
    def test_pooled_mean_std_returns_double_hidden_dim(self):
        hidden = torch.tensor([
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]],
        ])

        pooled = pooled_mean_std(hidden)

        self.assertEqual(pooled.shape, (2, 6))
        self.assertTrue(torch.allclose(pooled[0, :3], torch.tensor([2.0, 3.0, 4.0])))

    def test_ssl_pooled_mlp_forward_shape(self):
        model = SSLPooledMLP(input_dim=12, hidden_dim=8, dropout=0.1)

        y = model(torch.randn(4, 12))

        self.assertEqual(y.shape, (4, 2))

    def test_cached_embedding_dataset_returns_metadata(self):
        cache = {
            "items": [{
                "file_id": "LA_T_0001",
                "path": "sample.flac",
                "label": 1,
                "system_id": "-",
                "embedding": torch.ones(6),
            }]
        }
        dataset = SSLEmbeddingDataset(cache)

        item = dataset[0]

        self.assertEqual(item["x"].shape, (6,))
        self.assertEqual(int(item["label"]), 1)
        self.assertEqual(item["file_id"], "LA_T_0001")
        self.assertEqual(item["system_id"], "-")

    def test_train_mean_std_normalizer_uses_train_statistics(self):
        train = torch.tensor([[1.0, 3.0], [3.0, 7.0]])
        dev = torch.tensor([[2.0, 5.0]])

        normalizer = TrainMeanStdNormalizer.fit(train)
        transformed_train = normalizer.transform(train)
        transformed_dev = normalizer.transform(dev)

        self.assertTrue(torch.allclose(transformed_train.mean(dim=0), torch.zeros(2)))
        self.assertTrue(torch.allclose(transformed_dev, torch.zeros(1, 2)))

    def test_class_weights_from_labels_balances_inverse_frequency(self):
        weights = class_weights_from_labels([0, 0, 0, 1])

        self.assertTrue(torch.allclose(weights, torch.tensor([2.0 / 3.0, 2.0])))

    def test_cache_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.pt"
            path.write_text("existing")

            with self.assertRaises(FileExistsError):
                ensure_cache_writable(path)

            self.assertEqual(ensure_cache_writable(path, overwrite=True), path)

    def test_phase3_config_validator_accepts_expected_defaults(self):
        config = {
            "track": "external-pretrained/applied",
            "model": {"type": "ssl_pooled_mlp", "input": "wavlm_pooled_mean_std", "input_dim": 1536},
            "ssl": {
                "encoder_name": "microsoft/wavlm-base-plus",
                "cache_representation": "pooled_mean_std",
                "hidden_state_source": "last_hidden_state",
                "external_pretraining": True,
            },
            "data": {"sample_rate": 16000, "num_samples": 64600, "splits": ["dev", "eval"]},
            "training": {"normalization": "train_mean_std", "loss": "weighted_cross_entropy"},
        }

        self.assertEqual(validate_config(config), [])

    def test_cache_payload_validator_checks_pooled_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "model": {"input_dim": 6},
                "ssl": {"encoder_name": "toy/encoder", "encoder_slug": "toy_encoder"},
                "cache": {"root": tmp},
            }
            path = Path(tmp) / "toy_encoder" / "train.pt"
            path.parent.mkdir(parents=True)
            save_cache(path, {
                "encoder_name": "toy/encoder",
                "encoder_revision": "abc123",
                "processor_name": "toy_processor",
                "transformers_version": "test",
                "hidden_state_source": "last_hidden_state",
                "cache_representation": "pooled_mean_std",
                "sample_rate": 16000,
                "num_samples": 64600,
                "torch_dtype": "float32",
                "cache_device": "cpu",
                "split": "train",
                "items": [{
                    "file_id": "sample",
                    "path": "sample.flac",
                    "label": 1,
                    "system_id": "-",
                    "embedding": torch.ones(6),
                }],
            })

            self.assertEqual(validate_cache_payload(config, "train"), [])


if __name__ == "__main__":
    unittest.main()

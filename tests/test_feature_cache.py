import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.dataset import Sample
from src.feature_cache import FeatureCache
from src.features import SR


class FeatureCacheTests(unittest.TestCase):
    def test_cache_reuses_existing_feature_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "sample.flac"
            sf.write(audio_path, np.zeros(SR, dtype=np.float32), SR)
            sample = Sample(
                file_id="LA_T_TEST",
                path=str(audio_path),
                label=1,
                system_id="-",
            )
            cache = FeatureCache(tmp_path / "cache")
            calls = {"count": 0}

            def feature_fn(audio, sr):
                calls["count"] += 1
                return np.ones((4, 3), dtype=np.float32)

            first = cache.load_or_extract(sample, "unit", feature_fn)
            second = cache.load_or_extract(sample, "unit", feature_fn)

            np.testing.assert_array_equal(first, second)
            self.assertEqual(calls["count"], 1)
            self.assertTrue(cache.path_for(sample, "unit").exists())
            self.assertTrue(cache.path_for(sample, "unit").with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from src.features import N_COEFF, SR, lfcc_frames, linear_filterbank


class FeatureTests(unittest.TestCase):
    def test_lfcc_frames_are_finite_frame_level_features(self):
        rng = np.random.default_rng(0)
        audio = rng.normal(0.0, 0.05, size=SR).astype(np.float32)

        frames = lfcc_frames(audio, SR)

        self.assertEqual(frames.ndim, 2)
        self.assertEqual(frames.shape[1], N_COEFF * 3)
        self.assertGreater(frames.shape[0], 0)
        self.assertTrue(np.isfinite(frames).all())
        self.assertEqual(frames.dtype, np.float32)

    def test_linear_filterbank_shape_and_nonnegative_weights(self):
        fbanks = linear_filterbank(sr=SR, n_fft=512, n_filters=70)

        self.assertEqual(fbanks.shape, (70, 257))
        self.assertTrue(np.all(fbanks >= 0.0))
        self.assertTrue(np.all(fbanks.sum(axis=1) > 0.0))


if __name__ == "__main__":
    unittest.main()

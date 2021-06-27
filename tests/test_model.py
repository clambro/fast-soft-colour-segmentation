import unittest
from tempfile import NamedTemporaryFile

import numpy as np
from mock import patch

import config
from model import FSCSModel


class TestFSCSModel(unittest.TestCase):
    def test_build_evaluate_model(self):
        """Test that it builds, compiles, and evaluates the loss without error."""
        fscs = FSCSModel()
        img_height = 2**(config.NUM_DOWNSAMPLE_LAYERS + 2)
        img_batch = np.random.random((2, img_height, img_height, 3))
        fscs.model.evaluate(img_batch, img_batch, verbose=0)

    def test_save_load_model(self):
        fscs1 = FSCSModel()
        with NamedTemporaryFile() as file:
            fscs1.save_weights(file.name)
            fscs2 = FSCSModel(file.name)
        for w1, w2 in zip(fscs1.model.weights, fscs2.model.weights):
            np.testing.assert_array_almost_equal(w1, w2)

    @patch('tensorflow.keras.Model.fit')
    def test_optimize_for_one_image(self, tf_patch):
        fscs = FSCSModel()
        img = np.random.random((101, 256, 3))  # Prime and power of 2 size to test resizing.
        channels = fscs.optimize_for_one_image(img)

        self.assertEqual(len(channels), config.NUM_REGIONS)
        reconstructed_img = np.sum(channels, axis=0)
        np.testing.assert_array_equal(img.shape, reconstructed_img.shape)
        np.testing.assert_array_compare(np.greater_equal, reconstructed_img, 0)
        np.testing.assert_array_compare(np.less_equal, reconstructed_img, 1)


if __name__ == '__main__':
    unittest.main()

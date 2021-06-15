import config
from model import FSCSModel
import numpy as np
from tempfile import NamedTemporaryFile
import unittest


class TestModel(unittest.TestCase):
    def test_build_predict_model(self):
        """Test that it builds and runs without error."""
        fscs = FSCSModel()
        img_shape = (1, 2**(config.NUM_SEGMENTS + 2), 2**(config.NUM_SEGMENTS + 2), 3)
        fscs.model.predict(np.random.random(img_shape))

    def test_save_load(self):
        fscs1 = FSCSModel()
        with NamedTemporaryFile() as file:
            fscs1.save_weights(file.name)
            fscs2 = FSCSModel(file.name)
        for w1, w2 in zip(fscs1.model.weights, fscs2.model.weights):
            np.testing.assert_array_almost_equal(w1, w2)


if __name__ == '__main__':
    unittest.main()

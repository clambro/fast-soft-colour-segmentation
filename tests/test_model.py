import config
from model import FSCSModel
import numpy as np
import unittest


class TestModel(unittest.TestCase):
    def test_build_predict_model(self):
        """Just test that it builds and runs without error."""
        f = FSCSModel()
        f.model.predict(np.random.random((1, 2**(config.NUM_SEGMENTS + 2), 2**(config.NUM_SEGMENTS + 2), 3)))


if __name__ == '__main__':
    unittest.main()

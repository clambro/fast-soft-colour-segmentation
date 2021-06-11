from model import FSCSModel
import unittest


class TestModel(unittest.TestCase):
    def test_build_model(self):
        """Just test that it builds without error."""
        FSCSModel()


if __name__ == '__main__':
    unittest.main()

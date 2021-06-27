import unittest

from scripts.optimize import main


class TestOptimize(unittest.TestCase):
    def test_bad_img_path(self):
        with self.assertRaises(FileNotFoundError) as err:
            main('', '', 1)
        self.assertIn('not found', str(err.exception))

    def test_bad_out_path(self):
        with self.assertRaises(NotADirectoryError) as err:
            main('tests/scripts/test_optimize.py', 'not/a/valid/directory', 1)
        self.assertIn('not valid or does not exist', str(err.exception))

    def test_bad_scale(self):
        with self.assertRaises(ValueError) as err:
            main('tests/scripts/test_optimize.py', 'src', 3)
        self.assertIn('The scale parameter must be in (0, 1]', str(err.exception))


if __name__ == '__main__':
    unittest.main()

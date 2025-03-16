import unittest
from data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        x_train, x_test, y_train, y_test = load_data()
        self.assertEqual(x_train.shape[1:], (28, 28, 1))
        self.assertEqual(x_test.shape[1:], (28, 28, 1))

if __name__ == '__main__':
    unittest.main()

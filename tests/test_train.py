import unittest
from train import train_model

class TestTrain(unittest.TestCase):
    def test_train_model(self):
        try:
            train_model()
        except Exception as e:
            self.fail(f"Train model failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()

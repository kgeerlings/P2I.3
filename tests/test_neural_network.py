import unittest
from neural_network import create_model

class TestNeuralNetwork(unittest.TestCase):
    def test_model_creation(self):
        model = create_model()
        self.assertEqual(len(model.layers), 6)  # Vérifie le nombre de couches
        self.assertEqual(model.input_shape, (None, 28, 28, 1))

if __name__ == '__main__':
    unittest.main()

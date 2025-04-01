import os
import gzip
import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image

def load_data():
    # Charger le dataset MNIST (chiffres)
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(fname="data/mnist.npz") 
    # marche pas parce que fname peut pas chercher un fichier local
    
# charge les données MNIST depuis le fichier local 'data/mnist.npz'
    with np.load("data/mnist.npz") as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']

    # Normalisation des images (0-255 => 0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Redimensionnement des images (28x28 => 28x28x1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, x_test, y_train, y_test




def load_data_emnist(data_dir='./data/gzip'):
    """Charge et retourne les données EMNIST normalisées avec correction d'orientation"""

    # Chemins des fichiers
    train_images_path = os.path.join(data_dir, 'emnist-balanced-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 'emnist-balanced-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')

    # Vérification des fichiers
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} est introuvable. Vérifie le chemin !")

    # Chargement des images
    with gzip.open(train_images_path, 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)
    with gzip.open(train_labels_path, 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(test_images_path, 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)
    with gzip.open(test_labels_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    # Correction d'orientation (les images sont mal tournées dans EMNIST)
    x_train = np.rot90(x_train, k=-1, axes=(1, 2))  # Rotation pour être aligné avec MNIST
    x_test = np.rot90(x_test, k=-1, axes=(1, 2))

    # Normalisation des images (valeurs entre 0 et 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Debugging
    print(f"x_train shape: {x_train.shape}")  # (N, 28, 28, 1)
    print(f"x_test shape: {x_test.shape}")    # (N, 28, 28, 1)
    print(f"y_train shape: {y_train.shape}") # (N, 47)
    print(f"y_test shape: {y_test.shape}")   # (N, 47)

    return x_train, x_test, y_train, y_test



import matplotlib as plt
import numpy as np
import gzip
import os

def load_raw_emnist(data_dir='./data/gzip'):
    """Charge les images et labels EMNIST sans normalisation ni transformation"""
    test_images_path = os.path.join(data_dir, 'emnist-balanced-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')

    with gzip.open(test_images_path, 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(test_labels_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    return x_test, y_test

def plot_emnist_samples(data_dir='./data/gzip', num_samples=10):
    """Affiche quelques images du dataset EMNIST avec leurs labels"""
    x_test, y_test = load_raw_emnist(data_dir)

    # Correction d'orientation
    x_test = np.rot90(x_test, k=-1, axes=(1, 2))  # Alignement avec MNIST

    # Labels EMNIST Balanced : Chiffres (0-9) + Lettres majuscules/minuscules fusionnées (A-Z)
    emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    
    plt.figure(figsize=(12, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"Label: {emnist_classes[y_test[i]]}")
        plt.axis('off')

    plt.show()

# 📌 Affichage des images
plot_emnist_samples()
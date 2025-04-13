import matplotlib.pyplot as plt
import numpy as np
import gzip
import os
from data_loader import load_data_emnist

import torch
from torchvision import datasets, transforms

# Définir une transformation (normalisation par exemple)
transform = transforms.Compose([transforms.ToTensor()])

# Charger le dataset EMNIST Balanced
train_data = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)

# Vérifier quelques labels
print("Quelques labels d'entraînement : ", train_data.targets[:10])


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

    # Correction d'orientation : retournement horizontal des images (miroir)
    x_test = np.fliplr(x_test)  # Applique le miroir horizontal pour corriger l'orientation
    x_test = np.rot90(x_test, k=-1, axes=(1, 2)) 

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
#plot_emnist_samples()

import matplotlib.pyplot as plt
import numpy as np
import gzip
import os

def load_raw_emnist(data_dir='./data/gzip'):
    """Charge les images et labels EMNIST sans normalisation ni transformation"""
    train_images_path = os.path.join(data_dir, 'emnist-balanced-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 'emnist-balanced-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')

    # Chargement des données
    with gzip.open(train_images_path, 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(train_labels_path, 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(test_images_path, 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(test_labels_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    return x_train, x_test, y_train, y_test

def plot_emnist_samples(x_data, y_data, emnist_classes, num_samples=10):
    """Affiche quelques images du dataset EMNIST avec leurs labels"""
    plt.figure(figsize=(12, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_data[i], cmap='gray')
        plt.title(f"Label: {emnist_classes[y_data[i]]}")
        plt.axis('off')

    plt.show()

# Charger les données
x_train, x_test, y_train, y_test = load_raw_emnist()

x_test = np.fliplr(x_test)
x_train = np.fliplr(x_train)
# Vérification de l'orientation : rotation de 90 degrés dans le sens des aiguilles d'une montre
x_train = np.rot90(x_train, k=-1, axes=(1, 2))  # Rotation pour x_train
x_test = np.rot90(x_test, k=-1, axes=(1, 2))    # Rotation pour x_test



# Afficher des exemples pour vérifier l'orientation
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
plot_emnist_samples(x_train, y_train, emnist_classes, num_samples=5)  # Afficher 5 exemples de x_train
plot_emnist_samples(x_test, y_test, emnist_classes, num_samples=5)   # Afficher 5 exemples de x_test

import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def load_data(data_dir='./data/gzip'):
    """Charge et retourne les données EMNIST normalisées"""
    train_images_path = os.path.join(data_dir, 'emnist-balanced-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 'emnist-balanced-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')

    with gzip.open(train_images_path, 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)
    with gzip.open(train_labels_path, 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(test_images_path, 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)
    with gzip.open(test_labels_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    return (x_train / 255.0, x_test / 255.0, y_train, y_test)

def build_model():
    """Construit et retourne un modèle CNN pour EMNIST"""
    model = keras.models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(47, activation='softmax')  # 47 classes pour EMNIST Balanced
    ])
    return model

def train2_model():
    """Entraîne le modèle sur EMNIST et sauvegarde les résultats"""
    x_train, x_test, y_train, y_test = load_data()

    import random

    # Vérifier plusieurs images et leurs labels
    for i in range(5):
        index = random.randint(0, len(x_test) - 1)
        plt.imshow(x_train[index].squeeze(), cmap="gray")
        plt.title(f"Label attendu: {y_train[index]}")
        plt.show()

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
    
    os.makedirs("models", exist_ok=True)
    model.save("models/emnist_model.h5")

    # Affichage et sauvegarde des résultats
    plot_training_history(history)
    plot_confusion_matrices(model, x_test, y_test)

def plot_confusion_matrices(model, x_test, y_test):
    """Affiche et sauvegarde les matrices de confusion pour lettres et chiffres"""
    y_pred = np.argmax(model.predict(x_test), axis=1)

    labels_digits = [str(i) for i in range(10)]  # Chiffres 0-9
    labels_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    labels_full = labels_digits + labels_letters  # Chiffres et lettres

    cm_full = confusion_matrix(y_test, y_pred)
    cm_digits = cm_full[:10, :10]  # Extraire uniquement la partie des chiffres
    cm_letters = cm_full[10:, 10:]  # Extraire uniquement la partie des lettres

    # Matrice de confusion complète
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_full, annot=False, fmt='d', cmap='Purples', xticklabels=labels_full, yticklabels=labels_full, square=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion - Ensemble complet")
    plt.savefig("models/confusion_matrix_full.png")
    plt.show()

    # Matrice de confusion pour les chiffres
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_digits, annot=True, fmt='d', cmap='Blues', xticklabels=labels_digits, yticklabels=labels_digits, square=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion - Chiffres")
    plt.savefig("models/confusion_matrix_digits.png")
    plt.show()

    # Matrice de confusion pour les lettres
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_letters, annot=True, fmt='d', cmap='Greens', xticklabels=labels_letters, yticklabels=labels_letters, square=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion - Lettres")
    plt.savefig("models/confusion_matrix_letters.png")
    plt.show()

def plot_training_history(history):
    """Affiche et sauvegarde les courbes de perte et d'exactitude"""
    plt.figure(figsize=(12, 5))
    
    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte entraînement')
    plt.plot(history.history['val_loss'], label='Perte validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()

    # Courbe d'exactitude
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Exactitude entraînement')
    plt.plot(history.history['val_accuracy'], label='Exactitude validation')
    plt.xlabel('Époques')
    plt.ylabel('Exactitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig("models/training_history.png")
    plt.show()

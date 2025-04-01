import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data_loader import load_data_emnist
from matrix import plot_confusion_matrices
from training_history import plot_training_history_emnist

def evaluate2_model():
    """Évalue le modèle et affiche les résultats"""
    # Charger les données
    x_train, x_test, y_train, y_test = load_data_emnist()

    #verification des données
    print(f"Taille x_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"Valeurs uniques y_test: {np.unique(y_test)}")

    plt.imshow(x_test[0].squeeze(), cmap="gray")
    plt.title(f"Label attendu: {y_test[0]}")
    plt.show()

    plt.imshow(x_train[0].squeeze(), cmap="gray")
    plt.title(f"Label attendu (train): {y_train[0]}")
    plt.show()

    # Vérifier le nombre d'éléments
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    # Vérifier si les labels sont bien répartis
    print("Labels train uniques:", np.unique(y_train))
    print("Labels test uniques:", np.unique(y_test))

    # Vérifier un alignement image-label
    plt.imshow(x_test[0].squeeze(), cmap="gray")
    plt.title(f"Label attendu: {y_test[0]}")
    plt.show()

    import random

    # Vérifier plusieurs images et leurs labels
    for _ in range(5):
        index = random.randint(0, len(x_test) - 1)
        plt.imshow(x_test[index].squeeze(), cmap="gray")
        plt.title(f"Label attendu: {y_test[index]}")
        plt.show()


    # Charger le modèle sauvegardé
    model = tf.keras.models.load_model("models/emnist_model.h5")

    # Évaluation du modèle
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Précision sur les données de test : {accuracy * 100:.2f}%")

    # Générer et afficher les matrices de confusion
    plot_confusion_matrices(model, x_test, y_test)

    # Afficher les courbes de perte et précision
    plot_training_history_emnist(model)

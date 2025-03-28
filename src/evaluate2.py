import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_data_emnist

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
    #plot_training_history(model)

# def plot_training_history(model):
#     """Affiche les courbes de perte et précision pour l'entraînement et la validation"""
#     # Charger les données
#     x_train, x_test, y_train, y_test = load_data_emnist()

#     # Compiler le modèle si nécessaire
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Tracer la courbe de perte
#     plt.figure(figsize=(12, 6))

#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Perte (Entraînement)', color='blue')
#     plt.plot(history.history['val_loss'], label='Perte (Validation)', color='red')
#     plt.title('Courbe de Perte')
#     plt.xlabel('Epochs')
#     plt.ylabel('Perte')
#     plt.legend()

#     # Tracer la courbe de précision
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Précision (Entraînement)', color='blue')
#     plt.plot(history.history['val_accuracy'], label='Précision (Validation)', color='red')
#     plt.title('Courbe de Précision')
#     plt.xlabel('Epochs')
#     plt.ylabel('Précision')
#     plt.legend()

#     # Afficher les courbes
#     plt.tight_layout()
#     plt.show()

# Fonction pour afficher les matrices de confusion
def plot_confusion_matrices(model, x_test, y_test):
    """Affiche et sauvegarde les matrices de confusion pour les lettres"""
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Labels pour affichage
    labels_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    labels_digits = [str(i) for i in range(10)]  # Chiffres 0-9
    labels_full = labels_digits + labels_letters  # Chiffres et lettres

    # Matrice de confusion pour les lettres (valeurs entre 0 et 25)
    cm_letters = confusion_matrix(y_test, y_pred)
    cm_full = confusion_matrix(y_test, y_pred)
    cm_digits = cm_full[:10, :10]  # Extraire uniquement la partie des chiffres

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_letters, annot=True, fmt='d', cmap='Purples', xticklabels=labels_letters, yticklabels=labels_letters, square=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion - Lettres")
    plt.savefig("models/confusion_matrix_letters_evaluate.png")
    plt.show()

#matrice pour les chiffres
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_digits, annot=True, fmt='d', cmap='Blues', xticklabels=labels_digits, yticklabels=labels_digits, square=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion - Chiffres")
    plt.savefig("models/confusion_matrix_digits_evaluate.png")
    plt.show()

    # Matrice de confusion complète
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_full, annot=False, fmt='d', cmap='Purples', xticklabels=labels_full, yticklabels=labels_full, square=True)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion - Ensemble complet")
    plt.savefig("models/confusion_matrix_full_evaluate.png")
    plt.show()

if __name__ == '__main__':
    evaluate2_model()

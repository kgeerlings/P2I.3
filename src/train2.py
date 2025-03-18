import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data_loader import load_data_emnist
from models import build_model_emnist


def train2_model():
    """Entraîne le modèle sur EMNIST et sauvegarde les résultats"""
    x_train, x_test, y_train, y_test = load_data_emnist()

    import random

    # Vérifier plusieurs images et leurs labels
    for i in range(5):
        index = random.randint(0, len(x_test) - 1)
        plt.imshow(x_train[index].squeeze(), cmap="gray")
        plt.title(f"Label attendu: {y_train[index]}")
        plt.show()

    model = build_model_emnist()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
    
    os.makedirs("models", exist_ok=True)
    model.save("models/emnist_model.h5")

    # Affichage et sauvegarde des résultats
    plot_confusion_matrices(model, x_test, y_test)




#création des matrices de confusion
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
    sns.heatmap(cm_letters, annot=True, fmt='d', cmap='Greens', 
            xticklabels=labels_letters, yticklabels=labels_letters, 
            square=True, cbar=True, linewidths=0.8, linecolor='black')
    
   # Ajout des labels et du titre
    plt.xlabel("Prédictions", fontsize=14)
    plt.ylabel("Vraies classes", fontsize=14)
    plt.title("Matrice de confusion - Lettres", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)

    # Sauvegarde et affichage
    plt.savefig("models/confusion_matrix_letters.png", bbox_inches='tight', dpi=300)
    plt.show()


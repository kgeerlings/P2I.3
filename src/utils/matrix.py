import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
import sys
import tensorflow as tf 


current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, ".."))  # remonte à /src
sys.path.append(src_dir)
from dataloader.data_loader import load_data_emnist

def load_emnist_labels(mapping_path):
    """Charge les labels EMNIST Balanced depuis le fichier de mapping"""
    labels = []
    with open(mapping_path, 'r') as f:
        for line in f:
            _, unicode_val = line.strip().split()
            labels.append(chr(int(unicode_val)))
    return labels

# matrice de confusion pour mnist (seulement les chiffres)
def plot_confusion_matrix_mnist(model, x_test, y_test):
    """ Affiche la matrice de confusion """
    
    # Prédictions du modèle
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convertir les probabilités en classes

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred_classes)

    # Affichage avec Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion")
    plt.savefig('models/mnist_matrix/confusion_matrix_mnist.png')
    plt.show()



# Mapping EMNIST Balanced (index -> char)
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def plot_confusion_matrices(model, x_test, y_test, mapping_path="data/gzip/emnist-balanced-mapping.txt"):
    """Affiche et sauvegarde les matrices de confusion pour EMNIST Balanced"""
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Chargement des vrais labels EMNIST
    labels_emnist = load_emnist_labels(mapping_path)
    num_classes = len(labels_emnist)

    # Création de la matrice de confusion complète
    cm_full = confusion_matrix(y_test, y_pred, labels=range(num_classes))

    # Indices des chiffres et des lettres dans le label set
    digits_idx = [i for i, c in enumerate(labels_emnist) if c.isdigit()]
    letters_idx = [i for i, c in enumerate(labels_emnist) if c.isalpha()]

    sns.set(style="white")

    # --- Matrice complète ---
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(cm_full, annot=False, cmap='coolwarm',
                     xticklabels=labels_emnist, yticklabels=labels_emnist,
                     square=True, cbar=True)
    ax.set_xticklabels(labels_emnist, rotation=90, ha='center', fontsize=7)
    ax.set_yticklabels(labels_emnist, rotation=0, va='center', fontsize=7)
    plt.xlabel("Label prédit", fontsize=12)
    plt.ylabel("Label réel", fontsize=12)
    plt.title("Matrice de confusion - Ensemble complet", fontsize=14)
    plt.savefig("models/emnist_matrix/confusion_matrix_full.png", bbox_inches='tight', dpi=300)
    plt.show()

    # --- Matrice chiffres ---
    cm_digits = cm_full[np.ix_(digits_idx, digits_idx)]
    labels_digits = [labels_emnist[i] for i in digits_idx]

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm_digits, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels_digits, yticklabels=labels_digits,
                     annot_kws={"size": 8}, square=True, cbar=False)
    plt.xlabel("Label prédit", fontsize=12)
    plt.ylabel("Label réel", fontsize=12)
    plt.title("Matrice de confusion - Chiffres", fontsize=14)
    plt.savefig("models/emnist_matrix/confusion_matrix_digits.png", bbox_inches='tight', dpi=300)
    plt.show()

    # --- Matrice lettres ---
    cm_letters = cm_full[np.ix_(letters_idx, letters_idx)]
    labels_letters = [labels_emnist[i] for i in letters_idx]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_letters, annot=True, fmt='d', cmap='YlGnBu',
                     xticklabels=labels_letters, yticklabels=labels_letters,
                     annot_kws={"size": 6}, square=True, cbar=True)
    ax.set_xticklabels(labels_letters, rotation=90, ha='center', fontsize=7)
    ax.set_yticklabels(labels_letters, rotation=0, va='center', fontsize=7)
    plt.xlabel("Label prédit", fontsize=12)
    plt.ylabel("Label réel", fontsize=12)
    plt.title("Matrice de confusion - Lettres", fontsize=14)
    plt.savefig("models/emnist_matrix/confusion_matrix_letters.png", bbox_inches='tight', dpi=300)
    plt.show()

#x_train, x_test, y_train, y_test = load_data_emnist()
#model = tf.keras.models.load_model("models/emnist_model_20250413_210913/model.h5")

#plot_confusion_matrices(model, x_test, y_test)
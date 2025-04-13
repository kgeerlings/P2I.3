import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


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






# matrices de confusion pour emnist (lettres, chiffres & complète)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(model, x_test, y_test):
    """Affiche et sauvegarde les matrices de confusion pour lettres et chiffres"""
    y_pred = np.argmax(model.predict(x_test), axis=1)

    labels_digits = [str(i) for i in range(10)]  # Chiffres 0-9
    labels_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # Lettres A-Z
    labels_full = labels_digits + labels_letters  # Chiffres et lettres

    cm_full = confusion_matrix(y_test, y_pred)
    cm_digits = cm_full[:10, :10]  # Partie des chiffres
    cm_letters = cm_full[10:, 10:]  # Partie des lettres

    sns.set(style="white")  # Amélioration du style

    # Matrice complète
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_full, annot=False, cmap='coolwarm', xticklabels=labels_full, yticklabels=labels_full, square=True, cbar=True)
    ax.set_xticks(np.arange(len(labels_full)) + 0.5)
    ax.set_yticks(np.arange(len(labels_full)) + 0.5)
    ax.set_xticklabels(labels_full, rotation=0, ha='center', fontsize=8)
    ax.set_yticklabels(labels_full, rotation=0, va='center', fontsize=8)
    plt.xlabel("Label prédit", fontsize=12)
    plt.ylabel("Label réel", fontsize=12)
    plt.title("Matrice de confusion - Ensemble complet", fontsize=14)
    plt.savefig("models/emnist_matrix/confusion_matrix_full.png", bbox_inches='tight', dpi=300)
    plt.show()

    # Matrice pour les chiffres
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm_digits, annot=True, fmt='d', cmap='Blues', xticklabels=labels_digits, yticklabels=labels_digits, annot_kws={"size": 8}, square=True, cbar=False)
    plt.xlabel("Label prédit", fontsize=12)
    plt.ylabel("Label réel", fontsize=12)
    plt.title("Matrice de confusion - Chiffres", fontsize=14)
    plt.savefig("models/emnist_matrix/confusion_matrix_digits.png", bbox_inches='tight', dpi=300)
    plt.show()

    # Matrice pour les lettres
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_letters, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels_letters, yticklabels=labels_letters, annot_kws={"size": 6}, square=True, cbar=True)
    ax.set_xticks(np.arange(len(labels_letters)) + 0.5)
    ax.set_yticks(np.arange(len(labels_letters)) + 0.5)
    ax.set_xticklabels(labels_letters, rotation=0, ha='center', fontsize=8)
    ax.set_yticklabels(labels_letters, rotation=0, va='center', fontsize=8)
    plt.xlabel("Label prédit", fontsize=12)
    plt.ylabel("Label réel", fontsize=12)
    plt.title("Matrice de confusion - Lettres", fontsize=14)
    plt.savefig("models/emnist_matrix/confusion_matrix_letters.png", bbox_inches='tight', dpi=300)
    plt.show()

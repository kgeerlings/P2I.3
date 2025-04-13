import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from dataloader.data_loader import load_data_emnist

def evaluate2_model():
    """Évalue le modèle et affiche les résultats"""
    # Charger les données
    x_train, x_test, y_train, y_test = load_data_emnist()

    # Mapping des labels EMNIST Balanced
    labels_mapping = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Chiffres
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Lettres majuscules
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v'  # Lettres minuscules
    ]

    # Vérification des dimensions des données
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"Valeurs uniques y_test: {np.unique(y_test)}")

    # Vérifier l’alignement image-label sur quelques exemples
    for i in range(5):
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(f"Index: {i} - Label attendu: {labels_mapping[y_test[i]]}")
        plt.axis("off")
        plt.show()
    
    # Charger le modèle sauvegardé
    "meilleur modèle"
    model = tf.keras.models.load_model("models/emnist_model_20250413_210913/model.h5")
    #model = tf.keras.models.load_model("models/emnist_model_20250413_213608/model.h5") 

    # Évaluation du modèle
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Précision sur les données de test : {accuracy * 100:.2f}%")

    # Générer et afficher la matrice de confusion
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=47).numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title("Matrice de confusion")
    plt.show()

    # Visualisation des prédictions avec les images et leurs vrais labels
    for i in range(5):
        index = random.randint(0, len(x_test) - 1)
        plt.imshow(x_test[index].squeeze(), cmap="gray")
        # Utiliser labels_mapping pour afficher les vrais labels et les prédictions sous forme de caractères
        # On ajuste le mappage pour que les labels minuscules n'aient pas de décalage
        plt.title(f"Prédiction: {labels_mapping[y_pred[index]]} - Vrai label: {labels_mapping[y_test[index]]}")
        plt.axis("off")
        plt.show()

    # Tracer les courbes d'entraînement
    history = model.history.history
    loss_values = history["loss"]
    val_loss_values = history["val_loss"]
    acc_values = history["accuracy"]
    val_acc_values = history["val_accuracy"]

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, "bo-", label="Perte entraînement")
    plt.plot(epochs, val_loss_values, "r*-", label="Perte validation")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_values, "bo-", label="Précision entraînement")
    plt.plot(epochs, val_acc_values, "r*-", label="Précision validation")
    plt.xlabel("Époques")
    plt.ylabel("Précision")
    plt.legend()

    plt.show()
    #model = tf.keras.models.load_model("models/emnist_model.h5")
if __name__ == "__main__":
    evaluate2_model()
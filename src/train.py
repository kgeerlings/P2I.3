import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from data_loader import load_data
from models import build_model

#entrainement du modele sur MNIST (seuelemtn sur les chiffres)
def train_model():
    x_train, x_test, y_train, y_test = load_data()

    # Normalisation des images
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Création et compilation du modèle
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Entraînement du modèle
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Sauvegarde du modèle
    model.save('models/my_model.keras')

    # Tracer les courbes d'entraînement
    plot_training_history(history)

    # Évaluation du modèle et affichage de la matrice de confusion
    plot_confusion_matrix(model, x_test, y_test)

def plot_confusion_matrix(model, x_test, y_test):
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
    plt.savefig('models/confusion_matrix_mnist.png')
    plt.show()

def plot_training_history(history):
    """ Affiche les courbes de perte et d'exactitude pendant l'entraînement """
    
    # Historique de la perte et de l'exactitude
    plt.figure(figsize=(12, 5))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Perte d\'entraînement vs validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()

    # Courbe d'exactitude
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Exactitude d\'entraînement')
    plt.plot(history.history['val_accuracy'], label='Exactitude de validation')
    plt.title('Exactitude d\'entraînement vs validation')
    plt.xlabel('Époques')
    plt.ylabel('Exactitude')
    plt.legend()

    # Affichage des courbes
    plt.tight_layout()
    plt.savefig('models/training_history_mnist.png')
    plt.show()

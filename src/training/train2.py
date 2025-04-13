import os
import matplotlib.pyplot as plt
from dataloader.data_loader import load_data_emnist
from models.models import build_model_emnist
from utils.matrix import plot_confusion_matrices
from utils.training_history import plot_training_history_emnist
import random

def train2_model():
    """Entraîne le modèle sur EMNIST et sauvegarde les résultats"""
    x_train, x_test, y_train, y_test = load_data_emnist()

    # Vérifier plusieurs images et leurs labels
    for i in range(5):
        index = random.randint(0, len(x_train) - 1)
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
    plot_training_history_emnist(history)

if __name__ == '__main__':  
    train2_model()
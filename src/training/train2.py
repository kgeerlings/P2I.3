import os
import random
import datetime
import matplotlib.pyplot as plt

from dataloader.data_loader import load_data_emnist
from models.models import build_model_emnist
from utils.matrix import plot_confusion_matrices
from utils.training_history import plot_training_history_emnist

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train2_model():
    """Entraîne le modèle EMNIST avec augmentation, early stopping, scheduler, etc."""

    # Chargement des données
    x_train, x_test, y_train, y_test = load_data_emnist()

    # Séparation explicite validation/train
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    # Aperçu des données
    for i in range(5):
        index = random.randint(0, len(x_train) - 1)
        plt.imshow(x_train[index].squeeze(), cmap="gray")
        plt.title(f"Label attendu : {y_train[index]}")
        plt.axis('off')
        plt.show()

    # Construction et compilation du modèle
    model = build_model_emnist()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    # Entraînement
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[early_stop, lr_scheduler],
        verbose=2
    )

    # Sauvegarde du modèle
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/emnist_model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.h5"))

    # Visualisation des résultats
    plot_confusion_matrices(model, x_test, y_test)
    plot_training_history_emnist(history)

    # Résumé
    print(f"Entraînement terminé. Modèle sauvegardé dans : {model_dir}")
    print("Nombre d'epochs utilisées :", len(history.history['loss']))

if __name__ == '__main__':
    train2_model()
 
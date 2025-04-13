from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def build_model():
    # Créer un modèle séquentiel
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Remplace cette taille par celle de tes images
        Dense(128, activation='relu'),  # Couche cachée
        Dense(10, activation='softmax')  # Couche de sortie pour 10 classes (par exemple, MNIST)
    ])
    return model

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam

def build_model_emnist():
    """Construit et retourne un modèle CNN optimisé pour EMNIST Balanced."""

    model = models.Sequential()

    # Bloc 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Bloc 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Bloc 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Densely connected layer
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    # Sortie
    model.add(layers.Dense(47, activation='softmax'))  # EMNIST Balanced a 47 classes

    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

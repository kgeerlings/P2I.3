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

def build_model_emnist():
    """Construit et retourne un modèle CNN pour EMNIST"""
    model = keras.models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.02), input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.02)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02)),
        layers.Dropout(0.5),  # Pour éviter l'overfitting
        layers.Dense(47, activation='softmax')  # 47 classes pour EMNIST Balanced
    ])
    return model
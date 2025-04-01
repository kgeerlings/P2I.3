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

def build_model_emnist():
    """Construit et retourne un modèle CNN pour EMNIST avec des ajustements pour améliorer la précision"""
    model = models.Sequential([
        # Première couche convolutive avec plus de filtres
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # il faut augmenter la dropout pour réduire le sur-apprentissage

        # Deuxième couche convolutive avec encore plus de filtres et régularisation L1 L2
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Dropout pour éviter l'overfitting

        # Couches entièrement connectées
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),  # Dropout plus important pour éviter l'overfitting
        layers.Dense(47, activation='softmax')  # 47 classes pour EMNIST Balanced
    ])
    
    # Compilation du modèle avec l'optimiseur Adam et un taux d'apprentissage ajusté
    model.compile(optimizer=Adam(learning_rate=0.0005),  # Taux d'apprentissage plus faible pour une convergence plus stable
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

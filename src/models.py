from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_model():
    # Créer un modèle séquentiel
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Remplace cette taille par celle de tes images
        Dense(128, activation='relu'),  # Couche cachée
        Dense(10, activation='softmax')  # Couche de sortie pour 10 classes (par exemple, MNIST)
    ])
    return model

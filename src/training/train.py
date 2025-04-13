from dataloader.data_loader import load_data
from models.models import build_model
from utils.matrix import plot_confusion_matrix_mnist
from utils.training_history import plot_training_history_mnist



# entrainement du modele sur MNIST (seuelemtn sur les chiffres)
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
    model.save('models/mnist_model.keras')
    model.save('models/mnist_model.h5')

    # Tracer les courbes d'entraînement
    plot_training_history_mnist(history)

    # Évaluation du modèle et affichage de la matrice de confusion
    plot_confusion_matrix_mnist(model, x_test, y_test)

if __name__ == '__main__':
    train_model()
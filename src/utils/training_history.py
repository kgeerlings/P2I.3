import matplotlib.pyplot as plt


def plot_training_history_mnist(history):
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
    plt.savefig('models/mnist_matrix/training_history_mnist.png')
    plt.show()


def plot_training_history_emnist(history):
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
    plt.savefig('models/emnist_matrix/training_history_emnist.png')
    plt.show()

    
import tensorflow as tf
from dataloader.data_loader import load_data

def evaluate_model():
    x_train, x_test, y_train, y_test = load_data()

    # Charger le modèle
    model = tf.keras.models.load_model('/Users/kamigeerlings/Documents/P2I.3/models/mnist_model.h5')


    # Évaluation
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy : {accuracy * 100:.2f}%")

if __name__ == '__main__':
    evaluate_model()

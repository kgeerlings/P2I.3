import sys
import tensorflow as tf
from PyQt6.QtWidgets import QApplication
from mon_appli import Ui_Dialog
from evaluate2 import evaluate2_model
from recognition_app import RecognitionApp  # Classe de l'interface
from data_loader import load_data_emnist

def evaluate2(model_path, test_data, test_labels):
    """Fonction de validation du modèle."""
    model = tf.keras.models.load_model('./models/emnist_model.h5')
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Validation - Perte: {loss:.4f}, Exactitude: {accuracy:.4f}")

    if accuracy > 0.90:  # Seulement si l'exactitude est suffisante
        print("Validation réussie, lancement de l'interface...")
        return model
    else:
        print("Validation échouée. L'interface ne s'affichera pas.")
        return None

if __name__ == "__main__":
    # Charger les données de test
    x_train, y_train, x_test, y_test = load_data_emnist()

    # Exécuter evaluate2() avant d'afficher l'interface
    model = evaluate2('./models/emnist_model.h5', x_test, y_test)

    if model:  # Lancer l'interface seulement si la validation réussit
        app = QApplication(sys.argv)
        window = RecognitionApp(model)
        window.show()
        sys.exit(app.exec_())
    else:
        print("Fin du programme.")

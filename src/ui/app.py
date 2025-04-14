import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from keras.models import load_model
from keras import backend as K

from PIL import Image, ImageOps
import scipy.ndimage

# Gestion de la compatibilité Pillow (ANTIALIAS → Resampling.LANCZOS)
try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = Image.ANTIALIAS


class ImageClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classification d'Image")
        self.setGeometry(100, 100, 500, 400)

        # Zone d'affichage de l'image
        self.graphicsView = QGraphicsView(self)
        self.graphicsView.setGeometry(50, 50, 200, 200)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Bouton d'importation d'image
        self.button = QPushButton("Importer une image", self)
        self.button.setGeometry(50, 270, 200, 40)
        self.button.clicked.connect(self.load_image)

        # Label de prédiction
        self.label = QLabel("Prédiction : ", self)
        self.label.setGeometry(50, 320, 400, 40)
        self.label.setAlignment(Qt.AlignCenter)

        # Charger le modèle
        self.model = load_model("models/emnist_model_20250413_210913/model.h5")

        # Charger les labels
        self.labels_mapping = self.load_emnist_labels("data/gzip/emnist-balanced-mapping.txt")

    def load_emnist_labels(self, mapping_path):
        labels = []
        try:
            with open(mapping_path, 'r') as f:
                for line in f:
                    _, unicode_val = line.strip().split()
                    labels.append(chr(int(unicode_val)))
        except Exception as e:
            print(f"Erreur lors du chargement des labels EMNIST : {e}")
        return labels

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            print(f"\nImage chargée : {file_path}")
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            self.scene.clear()
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
            self.predict_image(file_path)

    def predict_image(self, file_path):
        try:
            print(f"--- Prédiction en cours pour : {file_path} ---")

            # Nettoyage de session
            K.clear_session()

            # Chargement de l'image en niveaux de gris
            image = Image.open(file_path).convert("L")

            # Inverser les couleurs : EMNIST = texte blanc sur fond noir
            image = ImageOps.invert(image)

            # Redimensionner à 20x20 avec conservation du ratio
            image.thumbnail((20, 20), resample)

            # Créer une image noire 28x28 et coller l'image redimensionnée au centre
            new_image = Image.new("L", (28, 28), 0)
            upper_left = ((28 - image.size[0]) // 2, (28 - image.size[1]) // 2)
            new_image.paste(image, upper_left)

            # Centrage selon le centre de gravité
            np_image = np.array(new_image)
            cy, cx = scipy.ndimage.center_of_mass(np_image)
            shiftx = int(np.round(14 - cx))
            shifty = int(np.round(14 - cy))
            shifted = scipy.ndimage.shift(np_image, shift=(shifty, shiftx), mode='constant', cval=0.0)

            # Normalisation
            shifted = shifted.astype('float32') / 255.0

            # Mise en forme pour la prédiction
            image_array = np.expand_dims(shifted, axis=(0, -1))  # (1, 28, 28, 1)

            print("Pixels [ligne 0] :", image_array[0, 0])

            # Prédiction
            prediction = self.model.predict(image_array)
            predicted_index = np.argmax(prediction[0])
            predicted_label = self.labels_mapping[predicted_index]

            print(f"Prédiction brute : {prediction}")
            print(f"Indice : {predicted_index} → Caractère : {predicted_label}")

            self.label.setText(f"Prédiction : {predicted_label}")
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            self.label.setText("Erreur de prédiction")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())

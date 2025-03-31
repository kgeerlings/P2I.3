import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

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
        self.label.setGeometry(50, 320, 300, 40)
        self.label.setAlignment(Qt.AlignCenter)
        
        # Charger le modèle de deep learning
        self.model = load_model("/Users/kami/2A/P2I.3/models/emnist_model.h5")


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            self.scene.clear()
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
            self.predict_image(file_path)

    def predict_image(self, file_path):
        # Charger l'image et la prétraiter
        image = Image.open(file_path).convert("L").resize((28, 28))  # Conversion en niveaux de gris (MNIST-like)
        image = img_to_array(image) / 255.0  # Normalisation
        image = np.expand_dims(image, axis=0)  # Ajouter batch dimension

        # Effectuer la prédiction
        prediction = self.model.predict(image)
        predicted_label = np.argmax(prediction, axis=1)[0]  # Classe prédite (avec la plus grande probabilité)
        
        # Afficher la prédiction
        self.label.setText(f"Prédiction : {predicted_label}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())

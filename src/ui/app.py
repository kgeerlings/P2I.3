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
        self.model = load_model("/Users/kamigeerlings/Documents/P2I.3/models/emnist_model.h5")

        # Mapping des labels EMNIST Balanced
        self.labels_mapping = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Chiffres
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Lettres majuscules
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v'  # Lettres minuscules
        ]

    def load_image(self):
        # Ouvrir la boîte de dialogue pour choisir un fichier image
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            print(f"Image chargée : {file_path}")
            # Afficher l'image dans la zone de prévisualisation
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            self.scene.clear()  # Effacer la scène précédente avant d'ajouter la nouvelle image
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
            # Prédire la classe de l'image chargée
            self.predict_image(file_path)

    def predict_image(self, file_path):
        try:
            # Charger l'image et la prétraiter
            image = Image.open(file_path).convert("L").resize((28, 28))  # Conversion en niveaux de gris (MNIST-like)
            
            # Vérification visuelle de l'image avant traitement
            image.show()  # Cela ouvrira l'image pour vérifier si elle est correcte

            image = img_to_array(image) / 255.0  # Normalisation
            image = np.expand_dims(image, axis=0)  # Ajouter la dimension du batch

            print(f"Image prétraitée et prête pour la prédiction : {image.shape}")

            # Effectuer la prédiction
            prediction = self.model.predict(image)
            print(f"Sortie brute du modèle : {prediction}")

            # Trouver l'indice du label prédit
            predicted_label_index = np.argmax(prediction, axis=1)[0]  # Classe prédite (avec la plus grande probabilité)
            print(f"Indice de la prédiction : {predicted_label_index}")
            
            # Utiliser le labels_mapping pour afficher la prédiction sous forme de caractère
            predicted_label = self.labels_mapping[predicted_label_index]

            # Mettre à jour le label de prédiction pour afficher la nouvelle prédiction
            self.label.setText(f"Prédiction : {predicted_label}")
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            self.label.setText("Erreur de prédiction")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())

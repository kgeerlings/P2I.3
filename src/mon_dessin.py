from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QPoint
import numpy as np
import cv2

class DrawingCanvas(QLabel):
    """Un QLabel modifié pour servir de canevas de dessin."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # Taille de la zone de dessin
        self.pixmap = QPixmap(self.size())  
        self.pixmap.fill(Qt.white)  # Fond blanc
        self.setPixmap(self.pixmap)
        self.pen = QPen(Qt.black, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # Stylo noir épais
        self.last_point = None

    def mousePressEvent(self, event):
        self.last_point = event.pos()  # Enregistre le point de départ

    def mouseMoveEvent(self, event):
        if self.last_point:
            painter = QPainter(self.pixmap)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.pos())  # Trace une ligne
            painter.end()
            self.setPixmap(self.pixmap)
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        self.last_point = None  # Arrête le dessin

    def clear(self):
        """Efface le dessin."""
        self.pixmap.fill(Qt.white)
        self.setPixmap(self.pixmap)

    def get_image_array(self):
        """Convertit le dessin en une image utilisable par le modèle."""
        image = self.pixmap.toImage()
        width, height = image.width(), image.height()
        data = image.bits().asstring(width * height * 4)  
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)  # Convertit en niveaux de gris
        resized = cv2.resize(gray_image, (28, 28))  # Ajuste à la taille du modèle
        normalized = resized / 255.0  # Normalisation
        return normalized.reshape(1, 28, 28, 1)

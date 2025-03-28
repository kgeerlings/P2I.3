from PyQt6.QtWidgets import QMainWindow, QVBoxLayout
from mon_appli import Ui_Dialog
from mon_dessin import DrawingCanvas  # Assurez-vous que cette classe existe

class RecognitionApp(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.ui = QMainWindow()
        self.ui.setupUi(self)
        self.model = model  

        # Ajouter le canvas de dessin
        self.canvas = DrawingCanvas(self)
        layout = QVBoxLayout(self.ui.drawingCanvas)
        layout.addWidget(self.canvas)

        # Connexion des boutons
        self.ui.clearButton.clicked.connect(self.canvas.clear)
        self.ui.predictButton.clicked.connect(self.predict_digit)

    def predict_digit(self):
        """Prédit le chiffre ou la lettre dessinée."""
        img = self.canvas.get_image_array()
        prediction = self.model.predict(img)
        predicted_class = prediction.argmax()
        self.ui.resultLabel.setText(f"Prédiction : {predicted_class}")

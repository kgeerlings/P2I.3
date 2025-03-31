from PyQt6.QtWidgets import QMainWindow, QVBoxLayout
from interface_ui import Ui_MainWindow
#from mon_dessin import DrawingCanvas  # Assurez-vous que cette classe existe

class RecognitionApp(QMainWindow):
    def __init__(self, model):
        super().__init__()

        # Crée une instance de la classe générée par pyuic6
        self.ui = Ui_MainWindow()  
        
        # Applique l'interface à cette fenêtre principale
        self.ui.setupUi(self)  

        self.model = model  

        # Ajouter le canvas de dessin
        #self.canvas = DrawingCanvas(self)
        
        # Ajoute le canvas au layout du widget de dessin (dépend de la structure de ton UI)
        #layout = QVBoxLayout(self.ui.drawingCanvas)  # Assurez-vous que `drawingCanvas` est un widget
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

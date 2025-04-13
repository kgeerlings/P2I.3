# Neural Network for Handwritten Character Recognition

Ce projet implémente un réseau de neurones capable d'identifier des lettres et des chiffres manuscrits. Il utilise les datasets MNIST (chiffres) et EMNIST (lettres).

## Structure du projet
- `data/`: Contient les datasets bruts et pré-traités. 
   C'est un dossier à créer vous-mëme car les taille des dataset est trop importante  
   pour les mettre sur Github. Voici les liens direct pour les installations de dataset:

   EMNIST: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip

- `models/`: Sauvegarde des modèles entraînés.
- `src/`: Code source du projet. Contient les entrainements, les évaluations, les courbes.
- `tests/`: Scripts de tests unitaires.
- `requirements.txt`: Dépendances Python à installer avec les bonnes versions.

## Installation
1. Clonez le dépôt.
2. Installez les dépendances :
   brew install python3.10
   brew install pyqt
   python3.10 -m pip install -r requirements.txt

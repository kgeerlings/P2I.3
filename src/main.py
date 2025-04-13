#from train import train_model
#from evaluate import evaluate_model
from evaluation.evaluate2 import evaluate2_model
from training.train2 import train2_model
from training.train import train_model
from evaluation.evaluate import evaluate_model
from ui.app import ImageClassifier

if __name__ == '__main__':
    print()
    print()
    print()
    print("1. Entraîner le modèle sur chiffres et lettres")
    print("2. Évaluer le smodèle sur chiffres et lettres")
    print("3. Entrainer le modèle sur chiffres")
    print("4. Evaluer le modèle sur chiffres")
    print("5. Lancer l'interface")
    print()
    print()
    choice = input("Choisissez une option : ")
    print()
    print()
    print()

    if choice == '1':
        train2_model()
    elif choice == '2':
        evaluate2_model()
    elif choice == '3':
        train_model()
    elif choice == '4':
        evaluate_model()
    elif choice == '5':
        ImageClassifier()
    else:
        print("Option invalide.")

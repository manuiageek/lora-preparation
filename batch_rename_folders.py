import os

# Fonction pour demander un chemin valide
def demander_chemin():
    while True:
        path = input("Veuillez entrer le chemin du répertoire contenant les dossiers : ").strip()
        if os.path.exists(path):
            return path
        print("Le chemin spécifié n'existe pas. Veuillez réessayer.")

# Fonction pour demander un index entier valide
def demander_index():
    while True:
        try:
            i = int(input("Veuillez entrer la valeur de départ pour l'index (ex: 1) : ").strip())
            return i
        except ValueError:
            print("La valeur de départ doit être un entier. Veuillez réessayer.")

# Demande à l'utilisateur les informations nécessaires
i = demander_index()
path = demander_chemin()

# Liste tous les dossiers dans le répertoire
for folder in sorted(os.listdir(path)):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        # Cherche un nouveau nom libre avec un padding de 2 chiffres
        new_name = str(i).zfill(2)
        new_folder_path = os.path.join(path, new_name)
        
        # Incrémente jusqu'à trouver un nom libre
        while os.path.exists(new_folder_path):
            i += 1
            new_name = str(i).zfill(2)
            new_folder_path = os.path.join(path, new_name)
        
        # Renomme le dossier
        os.rename(folder_path, new_folder_path)
        i += 1  # Passe au numéro suivant

print("Renommage des dossiers terminé.")

import os

# Définit le chemin vers le répertoire contenant les dossiers
path = r"F:\1_TO_EXTRACT_1-2-3\Sailor Moon\_-Eternal Movie"
i = 40

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

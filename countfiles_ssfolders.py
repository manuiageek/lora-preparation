import os


# Remplace "/chemin/vers/repertoire" par le chemin du répertoire que tu veux analyser
base_directory = r"F:\1_TO_EXTRACT\LOVE AFTER WORLD DOMINATION"

def count_files_in_subdirectories(base_directory):
    # Parcourir les sous-dossiers du répertoire de base
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            # Chemin complet du sous-dossier
            dir_path = os.path.join(root, dir_name)
            # Compter le nombre de fichiers dans ce sous-dossier
            num_files = sum([len(files) for _, _, files in os.walk(dir_path)])
            print(f"Le sous-dossier '{dir_name}' contient {num_files} fichier(s).")


count_files_in_subdirectories(base_directory)

import os


# Chemin du répertoire où se trouvent les fichiers
directory      = r"F:\1_TO_EXTRACT_1-2\DRAGON RAJA"

# Obtenir la liste des fichiers .mkv dans le répertoire
files = sorted([f for f in os.listdir(directory) if f.endswith('.mkv')])

# Boucle pour renommer les fichiers
for index, file in enumerate(files, start=1):
    # Créer un nouveau nom de fichier avec un format comme 01.mkv, 02.mkv, etc.
    new_name = f"{index:02d}.mkv"
    
    # Chemin complet pour l'ancien et le nouveau fichier
    old_file_path = os.path.join(directory, file)
    new_file_path = os.path.join(directory, new_name)
    
    # Renommer le fichier
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {file} -> {new_name}")

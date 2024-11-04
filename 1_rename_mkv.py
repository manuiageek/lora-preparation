import os





# Spécifier le répertoire de base
directory = r"F:\1_TO_EXTRACT_1-2-3\Sailor Moon\Eternal Movie"

# Obtenir la liste des fichiers .mkv et .mp4 dans le répertoire
files = sorted([f for f in os.listdir(directory) if f.endswith(('.mkv', '.mp4'))])

# Boucle pour renommer les fichiers
for index, file in enumerate(files, start=1):
    # Extraire l'extension du fichier (mkv ou mp4)
    file_extension = os.path.splitext(file)[1]
    
    # Créer un nouveau nom de fichier avec un format comme 01.mkv, 02.mp4, etc.
    new_name = f"{index:02d}{file_extension}"
    
    # Chemin complet pour l'ancien et le nouveau fichier
    old_file_path = os.path.join(directory, file)
    new_file_path = os.path.join(directory, new_name)
    
    # Renommer le fichier
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {file} -> {new_name}")

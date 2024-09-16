import os
import subprocess

# Spécifier le répertoire de base
base_directory = r"G:\TRAIN_LORA\_EXTRACT_VIDEO\_TO_EXTR\[Anime Time] Black Lagoon (Complete Series) (Season 01+02+03+OST) [BD] [Dual Audio] [1080p][HEVC 10bit x265][AAC][Eng Sub]\Season 3 Roberta's Blood Trail"

# Obtenir la liste de tous les fichiers .mkv dans le répertoire de base
mkv_files = [f for f in os.listdir(base_directory) if f.endswith('.mkv')]

for file in mkv_files:
    # Chemin complet vers le fichier
    file_path = os.path.join(base_directory, file)
    
    # Créer un dossier portant le même nom que le fichier (sans l'extension .mkv)
    dir_name = file[:-4]
    output_directory = os.path.join(base_directory, dir_name)
    os.makedirs(output_directory, exist_ok=True)

    # Utilisation de subprocess pour exécuter ffmpeg
    output_pattern = os.path.join(output_directory, 'frame_%08d.jpg')
    command = [
        'ffmpeg', '-i', file_path,
        '-vf', 'fps=15,select=not(mod(n\\,5))',
        '-q:v', '1', '-fps_mode', 'vfr',
        output_pattern
    ]
    
    # Exécuter la commande
    subprocess.run(command)

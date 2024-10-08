import os
import subprocess

# Spécifier le répertoire de base
base_directory = r"F:\1_TO_EXTRACT_1-2-3\[SubsPlease] Jaku-Chara Tomozaki-kun (01-12) (1080p) [Batch]"

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

    # Utilisation de subprocess pour exécuter ffmpeg (toutes les images !)    
#    output_pattern = os.path.join(output_directory, 'frame_%08d.jpg')
#    command = [
#        'ffmpeg', '-i', file_path,
#        '-q:v', '1',
#    output_pattern
#    ]

    
    # Exécuter la commande
    subprocess.run(command)

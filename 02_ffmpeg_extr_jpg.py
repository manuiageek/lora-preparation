import os
import subprocess

# Spécifier le répertoire de base
base_directory = "/media/hleet_user/HDD-EXT/[BlurayDesuYo] Full Metal Panic! Invisible Victory - BD BOX 02 (BD 1920x1080 10bit FLAC)"

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

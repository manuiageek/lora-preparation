import os
import subprocess




# Spécifier le répertoire de base
directory = r"F:\1_TO_EXTRACT_1-2-3\SPAWN\Todd.McFarlanes.Spawn.S03.COMPLETE.720p.HMAX.WEBRip.x264-GalaxyTV[TGx]"

# Obtenir la liste de tous les fichiers .mkv et .mp4 dans le répertoire de base
video_files = sorted(
    [f for f in os.listdir(directory) if f.endswith(('.mkv', '.mp4'))], 
    key=lambda f: os.path.getmtime(os.path.join(directory, f))
)

for file in video_files:
    # Chemin complet vers le fichier
    file_path = os.path.join(directory, file)
    
    # Créer un dossier portant le même nom que le fichier (sans l'extension)
    dir_name = os.path.splitext(file)[0]
    output_directory = os.path.join(directory, dir_name)
    os.makedirs(output_directory, exist_ok=True)

    # Utilisation de subprocess pour exécuter ffmpeg
    output_pattern = os.path.join(output_directory, 'frame_%08d.jpg')
    command = [
        'ffmpeg', '-i', file_path,
        '-vf', 'fps=18,select=not(mod(n\\,5))',
        '-q:v', '1', '-fps_mode', 'vfr',
        output_pattern
    ]

    # Utilisation de subprocess pour exécuter ffmpeg (toutes les images !)    
    # output_pattern = os.path.join(output_directory, 'frame_%08d.jpg')
    # command = [
    #     'ffmpeg', '-i', file_path,
    #     '-q:v', '1',
    #     output_pattern
    # ]

    # Exécuter la commande
    subprocess.run(command)

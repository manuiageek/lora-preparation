import os
import subprocess
import argparse
from datetime import datetime

# Fonction principale pour extraire des images à partir de vidéos
def extract_frames(directory):
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
            '-vf', 'fps=18,select=not(mod(n\,5))',
            '-q:v', '1', '-fps_mode', 'vfr',
            output_pattern
        ]

        # Exécuter la commande
        subprocess.run(command)
        
        # Afficher l'heure à laquelle le traitement est terminé
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Le fichier {file} a été traité avec succès à {end_time}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraire des images à partir de fichiers vidéo dans un répertoire donné.")
    parser.add_argument(
        "--directory",
        type=str,
        default=r"F:\\1_TO_EXTRACT_1-2-3\\SPAWN\\Todd.McFarlanes.Spawn.S03.COMPLETE.720p.HMAX.WEBRip.x264-GalaxyTV[TGx]",
        help="Le chemin du répertoire contenant les fichiers vidéo."
    )

    args = parser.parse_args()
    
    # Appeler la fonction principale
    extract_frames(args.directory)

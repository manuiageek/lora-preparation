import subprocess
from datetime import datetime
import sys

# Lire les paramètres à partir du fichier automate_5-3.txt
try:
    with open("automate_3-7.txt", "r", encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print("Erreur : Le fichier 'automate_3-7.txt' n'existe pas.")
    sys.exit(1)

# Boucle pour traiter chaque ligne du fichier
for line in lines:
    # Séparer la ligne en root_folder et character_file
    parts = line.split(';')
    if len(parts) != 2:
        print(f"Erreur : Ligne mal formatée : {line}")
        continue

    root_folder = parts[0]
    character_file = parts[1]

    # Exécuter le script 3_delete_duplicate_images.py
    script1 = "3_delete_duplicate_images.py"
    try:
        subprocess.run(
            ["python", script1, "--directory", root_folder],
            check=True
        )
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Le script {script1} a été exécuté avec succès pour le répertoire {root_folder} à {end_time}.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script {script1} pour le répertoire {root_folder} : {e}")    

    # Exécuter le script 6_TF_character_categorisation.py
    script2 = "7_TF_character_categorisation.py"
    try:
        subprocess.run(
            ["python", script2, "--root_folder", root_folder, "--character_file", character_file],
            check=True
        )
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Le script {script2} a été exécuté avec succès pour le répertoire {root_folder} à {end_time}.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script {script2} pour le répertoire {root_folder} : {e}")
        continue  # Passer à la ligne suivante en cas d'erreur

    # Exécuter le script 3_delete_duplicate_images.py
    try:
        subprocess.run(
            ["python", script1, "--directory", root_folder],
            check=True
        )
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Le script {script1} a été exécuté avec succès pour le répertoire {root_folder} à {end_time}.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script {script1} pour le répertoire {root_folder} : {e}") 

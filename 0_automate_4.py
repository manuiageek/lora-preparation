import subprocess
from datetime import datetime

# Nom du script à exécuter
script = "4_PT_detect_person_or_delete.py"

# Lire les répertoires à partir du fichier automate_4.txt
try:
    with open("automate_4.txt", "r") as file:
        directories = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print("Erreur : Le fichier 'automate_4.txt' n'existe pas.")
    exit(1)

# Boucle pour traiter chaque répertoire lu dans le fichier
for directory in directories:
    try:
        # Appeler le script avec le répertoire comme paramètre
        subprocess.run(["python", script, "--directory", directory], check=True)
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Le script {script} a été exécuté avec succès pour le répertoire {directory} à {end_time}.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script {script} pour le répertoire {directory} : {e}")

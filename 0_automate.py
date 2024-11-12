import subprocess

# Spécifier les chemins des scripts
scripts = [
    "1_rename_mkv.py",
    "2_ffmpeg_extr_jpg.py",
    "3_delete_duplicate_images.py"
]

# Lire les répertoires à partir du fichier automate_this.txt
try:
    with open("automate_this.txt", "r") as file:
        directories = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print("Erreur : Le fichier 'automate_this.txt' n'existe pas.")
    exit(1)

# Boucle pour traiter chaque répertoire lu dans le fichier
for directory in directories:
    for script in scripts:
        try:
            # Appeler chaque script avec le répertoire comme paramètre
            subprocess.run(["python", script, "--directory", directory], check=True)
            from datetime import datetime
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Le script {script} a été exécuté avec succès pour le répertoire {directory} à {end_time}.")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script {script} pour le répertoire {directory} : {e}")

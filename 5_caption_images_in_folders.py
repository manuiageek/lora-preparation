import os
import json
import copy
import requests

# ---------------------------
# Paramètres de l'API ComfyUI
# ---------------------------
API_URL = "http://127.0.0.1:8188/prompt"  # Vérifie que l'endpoint est correct
HEADERS = {"Content-Type": "application/json"}
WORKFLOW_FILE = "CAPTION_API.json"  # Chemin vers le fichier JSON du workflow

# ---------------------------
# Chemin et extensions
# ---------------------------
BASE_DIR = r"E:\AI_WORK\TRAINED_LORA\FAIRY TAIL 100 YEARS QUEST\_TEST"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif', '.heic'}

def list_images_from_ref_dirs(base_directory):
    """
    Parcourt le répertoire de base et recherche, pour chaque sous-dossier,
    le dossier 'ref' dans lequel il y a des images.
    """
    image_paths = []
    for entry in os.listdir(base_directory):
        entry_path = os.path.join(base_directory, entry)
        if os.path.isdir(entry_path):
            ref_dir = os.path.join(entry_path, "ref")
            if os.path.isdir(ref_dir):
                for filename in os.listdir(ref_dir):
                    file_path = os.path.join(ref_dir, filename)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename)
                        if ext.lower() in ALLOWED_EXTENSIONS:
                            image_paths.append(file_path)
    return image_paths

def call_api_for_image(image_path, workflow_template):
    """
    Pour une image donnée, la fonction crée une copie du workflow,
    modifie les paramètres relatifs au chemin de l'image, au dossier 'ref',
    et au nom du dossier parent du répertoire "ref", puis appelle l'API ComfyUI.
    Le JSON de réponse est renvoyé.
    """
    # Création d'une copie du workflow pour réinitialiser l'état à chaque itération
    workflow = copy.deepcopy(workflow_template)
    
    # Modification des paramètres du workflow :
    # Exemple de TAGS_EXCLUDED_PARAM (peut rester en dur)
    workflow["105"]["inputs"]["Text"] = "1girl,1boy,solo,"
    
    # Pour la clé KEYWORD_PARAM, on récupère le nom du dossier parent de "ref"
    # image_path = "...\\nom_dossier\\ref\\nom_image.ext"
    # os.path.dirname(image_path) retourne le chemin "...\\nom_dossier\\ref"
    # os.path.dirname(os.path.dirname(image_path)) retourne le chemin "...\\nom_dossier"
    # os.path.basename(...) récupère "nom_dossier"
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    workflow["67"]["inputs"]["Text"] = folder_name
    
    # PATH_IMG_PARAM : on insère le chemin complet de l'image trouvée
    workflow["94"]["inputs"]["value"] = image_path
    # PATH_PARAM : on insère le dossier "ref" de l'image 
    workflow["103"]["inputs"]["value"] = os.path.dirname(image_path)
    
    # Préparer le payload et envoyer la requête POST
    payload = {"prompt": workflow}
    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel pour l'image {image_path}: {e}")
        return None

if __name__ == "__main__":
    # Charger le template du workflow une seule fois
    try:
        with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
            workflow_template = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de {WORKFLOW_FILE}: {e}")
        exit(1)

    # Lister les images présentes dans tous les sous-dossiers 'ref'
    images = list_images_from_ref_dirs(BASE_DIR)
    print("Nombre d'images trouvées :", len(images))
    
    # Pour chaque image, appeler l'API et afficher le retour
    for image in images:
        print("Traitement de l'image :", image)
        result = call_api_for_image(image, workflow_template)
        if result is not None:
            print("Réponse de l'API :", result)
        else:
            print("Aucune réponse obtenue pour cette image.")
import os
import json
import copy
import requests
from openai import OpenAI  # Importation de la librairie OpenAI

# ---------------------------
# Paramètres de l'API ComfyUI
# ---------------------------
API_URL = "http://127.0.0.1:8187/prompt"  # Vérifie que l'endpoint est correct
HEADERS = {"Content-Type": "application/json"}
WORKFLOW_FILE = "CAPTION_API.json"      # Chemin vers le fichier JSON du workflow

# ---------------------------
# Extensions autorisées pour les images
# ---------------------------
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif', '.heic'}
ALLOWED_TXT_EXTENSIONS = {'.txt'}

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

def list_txt_files_from_ref_dirs(base_directory):
    """
    Parcourt le répertoire de base et recherche, pour chaque sous-dossier,
    le dossier 'ref' dans lequel il y a des fichiers .txt générés.
    """
    txt_paths = []
    for entry in os.listdir(base_directory):
        entry_path = os.path.join(base_directory, entry)
        if os.path.isdir(entry_path):
            ref_dir = os.path.join(entry_path, "ref")
            if os.path.isdir(ref_dir):
                for filename in os.listdir(ref_dir):
                    file_path = os.path.join(ref_dir, filename)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename)
                        if ext.lower() in ALLOWED_TXT_EXTENSIONS:
                            txt_paths.append(file_path)
    return txt_paths

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
    workflow["105"]["inputs"]["Text"] = "1girl,1boy,solo,breasts,large_breasts,medium_breasts,cleavage,between_breasts,small_breasts,looking_at_viewer,"
    
    # Pour la clé KEYWORD_PARAM, on récupère le nom du dossier parent de "ref"
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

def process_images():
    """
    Traite les images en parcourant le dossier de base pour trouver les images dans les sous-dossiers 'ref'.
    Pour chaque image, appelle l'API ComfyUI et affiche la réponse.
    Renvoie le dossier de base utilisé pour pouvoir le réutiliser ensuite.
    """
    while True:
        BASE_DIR = input("Veuillez entrer le chemin complet du dossier de base pour le traitement des images : ").strip()
        if os.path.isdir(BASE_DIR):
            break
        else:
            print("Le chemin saisi n'est pas un dossier valide. Veuillez réessayer.")

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
            print("Réponse de l'API ComfyUI :", result)
        else:
            print("Aucune réponse obtenue pour cette image.")
    
    # On retourne le dossier de base pour le traitement ultérieur des fichiers .txt
    return BASE_DIR

def process_caption_txt_with_openai(base_directory):
    """
    Pour chaque fichier .txt généré dans les dossiers 'ref' du dossier de base,
    lit son contenu, l'envoie à l'API OpenAI avec un pré-prompt précis, puis
    affiche la réponse qui est renvoyée et stocke le résultat dans 'user_keywords'.
    """
    # Charger la clé API depuis le fichier 'open_configkey.txt'
    try:
        with open("open_configkey.txt", "r", encoding="utf-8") as key_file:
            openaikey = key_file.read().strip()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier 'open_configkey.txt': {e}")
        return

    # Création du client OpenAI
    client = OpenAI(api_key=openaikey)

    # Pré-prompt imposé
    pre_prompt = (
        "I will provide you with a list of keywords. Reorder the keywords according to these rules:\n"
        "1) The first keyword remains in its original position.\n"
        "2) Next, include all keywords that describe physical attributes (e.g., eyes, hair).\n"
        "3) Then, include the keywords related to clothing.\n"
        "4) Finally, append all remaining keywords, including behavioral descriptors, at the end.\n"
        "Ensure that no keywords are omitted and that the output maintains the exact same format "
        "as the input without any additional explanations."
    )

    # Récupération de tous les fichiers .txt dans les dossiers 'ref'
    txt_files = list_txt_files_from_ref_dirs(base_directory)
    print("Nombre de fichiers .txt trouvés :", len(txt_files))
    
    # Pour chaque fichier texte, lire le contenu et appeler l'API OpenAI
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                user_keywords = f.read().strip()
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {txt_file}: {e}")
            continue

        print(f"\nTraitement du fichier : {txt_file}")
        print("Contenu du fichier (user_keywords) :", user_keywords)
        try:
            # Appel à l'API Chat de OpenAI
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": pre_prompt},
                    {"role": "user", "content": user_keywords}
                ]
            )
            # Extraire la réponse texte
            openai_response = completion.choices[0].message.content
            # La réponse est désormais assignée à user_keywords pour réutilisation éventuelle
            user_keywords = openai_response
            print("Réponse de l'API OpenAI :", user_keywords)
        except Exception as e:
            print(f"Une erreur s'est produite lors de l'appel à l'API OpenAI pour le fichier {txt_file} : {str(e)}")

def main():
    # Traiter les images avec l'API ComfyUI et récupérer le dossier de base utilisé
    base_dir = process_images()
    
    # Une fois les fichiers .txt générés dans les dossiers 'ref', on les envoie à OpenAI
    process_caption_txt_with_openai(base_dir)

if __name__ == "__main__":
    main()
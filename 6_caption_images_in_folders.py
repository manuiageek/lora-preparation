import os
import json
import copy
import sys
import time
import requests
from openai import OpenAI  # Importation de la librairie OpenAI

# ---------------------------
# Paramètres de l'API ComfyUI
# ---------------------------
API_URL = "http://127.0.0.1:8187/prompt"  # Vérifie que l'endpoint est correct
HEADERS = {"Content-Type": "application/json"}
WORKFLOW_FILE = "CAPTION_API.json"      # Chemin vers le fichier JSON du workflow

# ---------------------------
# Extensions autorisées pour les images et les fichiers texte
# ---------------------------
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif', '.heic'}
ALLOWED_TXT_EXTENSIONS = {'.txt'}

def call_llm(user_keywords, openai_client):
    """
    Appelle l'API LLM d'OpenAI en utilisant un mécanisme de retry.
    En cas d'échec après plusieurs tentatives, le script s'arrête avec un message d'erreur.
    """
    pre_prompt = (
        "I will provide you with a list of keywords. Reorder the keywords according to these rules:\n"
        "1) The first keyword remains in its original position.\n"
        "2) Next, include all keywords that describe physical attributes (e.g., eyes, hair colors).\n"
        "3) Then, exclude the keywords related to clothing.\n"
        "4) Finally, exclude all remaining keywords about behavioral description (e.g. own_hands_together, background).\n"
        "Please don't modify any keyword, leave them as they are.\n"
        "Ensure that the output maintains the exact same format "
        "as the input without any additional explanations. only one line output."
    )

    max_retries = 5         # Nombre maximum de tentatives
    retry_delay = 5         # Délai (en secondes) entre chaque tentative
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Tentative {attempt}/{max_retries} pour appeler OpenAI LLM...")
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": pre_prompt},
                    {"role": "user", "content": user_keywords}
                ],
                timeout=60  # Ajout d'un délai d'attente (60 sec) pour la requête
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Erreur lors de l'appel à OpenAI (tentative {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
            else:
                error_message = (
                    "\n****************************************************************\n"
                    "ERREUR FATALE : Impossible de contacter l'API OpenAI après plusieurs tentatives.\n"
                    "Vérifiez votre connexion internet, la validité de votre clé API, et l'état du service.\n"
                    "****************************************************************\n"
                )
                sys.exit(error_message)

def list_images_from_subfolders(base_directory):
    """
    Parcourt le répertoire de base et recherche, pour chaque sous-dossier,
    toutes les images directement présentes dans ce sous-dossier.
    """
    image_paths = []
    for entry in os.listdir(base_directory):
        entry_path = os.path.join(base_directory, entry)
        if os.path.isdir(entry_path):
            for filename in os.listdir(entry_path):
                file_path = os.path.join(entry_path, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in ALLOWED_EXTENSIONS:
                        image_paths.append(file_path)
    return image_paths

def list_txt_files_from_subfolders(base_directory):
    """
    Parcourt le répertoire de base et recherche, pour chaque sous-dossier,
    tous les fichiers .txt présents directement dans ce sous-dossier.
    """
    txt_paths = []
    for entry in os.listdir(base_directory):
        entry_path = os.path.join(base_directory, entry)
        if os.path.isdir(entry_path):
            for filename in os.listdir(entry_path):
                file_path = os.path.join(entry_path, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in ALLOWED_TXT_EXTENSIONS:
                        txt_paths.append(file_path)
    return txt_paths

def call_api_for_image(image_path, workflow_template):
    """
    Pour une image donnée, cette fonction crée une copie du workflow,
    modifie les paramètres relatifs au chemin de l'image, au dossier de l'image,
    et au nom du dossier parent (le nom du sous-dossier), puis appelle l'API ComfyUI.
    Le JSON de réponse est renvoyé.
    """
    workflow = copy.deepcopy(workflow_template)
    
    workflow["105"]["inputs"]["Text"] = (
        "1girl,1boy,solo,breasts,large_breasts,medium_breasts,"
        "cleavage,between_breasts,small_breasts,anime_coloring,looking_at_viewer,"
        "simple_background,white_background,smile,smiling,upper_body,personal_background,"
        "white_background,blue_sky,outdoors,blurry,sky,thumbs_up,smirk,closed_eyes,"
        "portrait,close-up,border,transparent_background,blurry_background,"
        "yellow_background,"
    )
    
    # Puisque les images se trouvent directement dans le sous-dossier, on récupère le nom du dossier parent
    folder_name = os.path.basename(os.path.dirname(image_path))
    workflow["67"]["inputs"]["Text"] = folder_name
    
    workflow["94"]["inputs"]["value"] = image_path
    workflow["103"]["inputs"]["value"] = os.path.dirname(image_path)
    
    payload = {"prompt": workflow}
    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel pour l'image {image_path}: {e}")
        return None

def parse_keywords(keywords_string):
    """
    Transforme une chaîne de mots-clés séparés par des virgules en un format recodé :
    '1er keyword':['keyword1','keyword2',...],
    où le premier mot-clé est utilisé comme clé et le reste comme liste de valeurs.
    """
    keywords = [kw.strip() for kw in keywords_string.split(",") if kw.strip()]
    
    if not keywords:
        return ""
    
    premier_keyword = keywords[0]
    autres = keywords[1:]
    autres_format = ",".join(f"'{mot}'" for mot in autres)
    output_line = f"'{premier_keyword}':[{autres_format}],\n"
    return output_line

def process_images():
    """
    Traite les images en parcourant le dossier de base pour trouver, 
    dans chacun de ses sous-dossiers, les images.
    Pour chaque image, appelle l'API ComfyUI et affiche la réponse.
    Renvoie le dossier de base utilisé pour pouvoir le réutiliser ensuite.
    """
    BASE_DIR = input("Veuillez entrer le chemin complet du dossier de base pour le traitement des images : ").strip()

    while not (BASE_DIR and os.path.isdir(BASE_DIR)):
        print("Vous devez saisir un chemin valide vers un dossier existant.")
        BASE_DIR = input("Veuillez entrer le chemin complet du dossier de base pour le traitement des images : ").strip()

    try:
        with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
            workflow_template = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de {WORKFLOW_FILE}: {e}")
        exit(1)

    images = list_images_from_subfolders(BASE_DIR)
    print("Nombre d'images trouvées :", len(images))
    
    for image in images:
        print("Traitement de l'image :", image)
        result = call_api_for_image(image, workflow_template)
        if result is not None:
            print("Réponse de l'API ComfyUI :", result)
        else:
            print("Aucune réponse obtenue pour cette image.")
    
    return BASE_DIR

def process_caption_txt_with_openai(base_directory):
    """
    Pour chaque fichier .txt présent dans les sous-dossiers du dossier de base,
    lit son contenu, l'envoie à l'API OpenAI avec un pré-prompt précis, puis
    reformule le résultat sous la forme :
        '1er keyword':['keyword1','keyword2',...],
    et écrit ce résultat dans un fichier CSV.
    Le fichier CSV est enregistré dans le répertoire "./chartags".
    """
    txt_files = list_txt_files_from_subfolders(base_directory)
    try:
        with open("open_configkey.txt", "r", encoding="utf-8") as key_file:
            openaikey = key_file.read().strip()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier 'open_configkey.txt': {e}")
        return

    client = OpenAI(api_key=openaikey)

    base_name = os.path.basename(os.path.normpath(base_directory))
    # Création du répertoire ./chartags s'il n'existe pas
    output_directory = os.path.join(".", "chartags")
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, base_name + ".csv")
    
    try:
        out_f = open(output_file, "w", encoding="utf-8")
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier {output_file} : {e}")
        return

    print("Attente de 1 secondes avant de débuter l'appel à l'API OpenAI...")
    time.sleep(1)

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                user_keywords = f.read().strip()
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {txt_file}: {e}")
            continue

        print(f"\nTraitement du fichier avec LLM : {txt_file}")
        try:
            response_text = call_llm(user_keywords, client)
        except Exception as e:
            print(f"Une erreur s'est produite lors de l'appel à l'API OpenAI pour le fichier {txt_file} : {str(e)}")
            continue

        formatted_output = parse_keywords(response_text)
        if formatted_output:
            try:
                out_f.write(formatted_output)
            except Exception as e:
                print(f"Erreur lors de l'écriture dans le fichier {output_file} : {e}")
    out_f.close()
    print(f"Les résultats OpenAI ont été enregistrés dans {output_file}")

def main():
    base_dir = process_images()
    process_caption_txt_with_openai(base_dir)

if __name__ == "__main__":
    main()

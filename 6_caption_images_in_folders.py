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

def call_llm(user_keywords, openai_client):
    """
    Appelle Ollama pour obtenir une réponse LLM. Si l'appel échoue,
    bascule sur l'API OpenAI.
    """
    # Pré-prompt imposé
    pre_prompt = (
        "I will provide you with a list of keywords. Reorder the keywords according to these rules:\n"
        "1) The first keyword remains in its original position.\n"
        "2) Next, include all keywords that describe physical attributes (e.g., eyes, hair colors).\n"
        "3) Then, delete the keywords related to clothing.\n"
        "4) Finally, delete all remaining keywords, including behavioral descriptors, at the end.\n"
        "Please don't modify any keyword, leave them as they are.\n"
        "Ensure that the output maintains the exact same format "
        "as the input without any additional explanations. only one line output."
    )

    ollama_url = "http://192.168.178.58:11434/v1/chat/completions"
    payload = {
        "model": "qwen2.5:7b-instruct-q5_K_M",
        "messages": [
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": user_keywords}
        ]
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=200)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print("Erreur lors de l'appel à Ollama, utilisation de OpenAI :", e)
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": user_keywords}
            ]
        )
        return completion.choices[0].message.content

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
    workflow["105"]["inputs"]["Text"] = "1girl,1boy,solo,breasts,large_breasts,medium_breasts,cleavage,between_breasts,small_breasts,anime_coloring,looking_at_viewer,"
    
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

def parse_keywords(keywords_string):
    """
    Transforme une chaîne de mots-clés séparés par des virgules en un format recodé :
    '1er keyword':['keyword1','keyword2',...],
    où le premier mot-clé est utilisé comme clé et le reste comme liste de valeurs.
    """
    # Séparation par la virgule et suppression des espaces superflus
    keywords = [kw.strip() for kw in keywords_string.split(",") if kw.strip()]
    
    if not keywords:
        return ""
    
    premier_keyword = keywords[0]
    autres = keywords[1:]
    # Construction de la chaîne selon le format souhaité
    # Exemple : '1er mot':['mot2','mot3',...],
    autres_format = ",".join(f"'{mot}'" for mot in autres)
    output_line = f"'{premier_keyword}':[{autres_format}],\n"
    return output_line

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
    reformule le résultat sous la forme :
        '1er keyword':['keyword1','keyword2',...],
    et écrit ce résultat dans un fichier CSV dont le nom est celui du dossier de base.
    Le fichier est ouvert en mode écriture.
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

    # Nom du répertoire de base (ajout de l'extension .csv)
    base_name = os.path.basename(os.path.normpath(base_directory))
    output_file = os.path.join(base_directory, base_name + ".csv")
    
    try:
        out_f = open(output_file, "w", encoding="utf-8")
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier {output_file} : {e}")
        return

    # Pour chaque fichier texte, lire le contenu et appeler l'API OpenAI, puis reformuler le résultat
    txt_files = list_txt_files_from_ref_dirs(base_directory)    
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
            print(f"Une erreur s'est produite lors de l'appel à l'API LLM pour le fichier {txt_file} : {str(e)}")
            continue

        formatted_output = parse_keywords(response_text)
        if formatted_output:
            try:
                out_f.write(formatted_output)
            except Exception as e:
                print(f"Erreur lors de l'écriture dans le fichier {output_file} : {e}")
    out_f.close()

def main():
    # Traiter les images avec l'API ComfyUI et récupérer le dossier de base utilisé
    base_dir = process_images()
    
    # Une fois les fichiers .txt générés dans les dossiers 'ref', on les envoie à OpenAI et écrit le résultat dans le fichier CSV
    process_caption_txt_with_openai(base_dir)

if __name__ == "__main__":
    main()
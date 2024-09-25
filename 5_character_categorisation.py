import deepdanbooru as dd
import numpy as np
from PIL import Image
import os
import glob
import shutil  # Pour gérer les fichiers et dossiers

# Spécifiez le chemin vers le projet DeepDanbooru (le dossier où vous avez extrait le modèle pré-entraîné)
project_path = './models/deepdanbooru'

# Charger le modèle
model = dd.project.load_model_from_project(project_path, compile_model=False)

# Charger les tags associés
tags = dd.project.load_tags_from_project(project_path)

# Convertir la liste des tags en un dictionnaire pour un accès plus rapide
tags_dict = {i: tag for i, tag in enumerate(tags)}

# Dictionnaire des personnages avec leurs caractéristiques (tags)
characters = {
    'akemi_dnm': ['bangs', 'black_hair', 'blunt_bangs', 'long_hair', 'blue_eyes'],
    'ayaka_dnm': ['brown_hair', 'bun', 'dark_skin', 'hair_behind_ear', 'hair_ornament', 'single_hair_bun', 'yellow_eyes'],
    'crystal_dnm': ['blue_eyes', 'long_hair', 'silver_hair'],
    'gina_dnm': ['blue_eyes', 'bob_cut', 'blunt_bangs', 'short_hair', 'white_hair'],
    'hibiki_dnm': ['blonde_hair', 'green_eyes', 'long_hair', 'twintails', 'bangs'],
    'satomi_dnm': ['black_hair', 'brown_eyes', 'short_hair']
}

# Fonction pour prétraiter l'image
def load_image_for_deepdanbooru(image_path, width, height):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image, dtype=np.float32) / 255.0
    return image.reshape((1, height, width, 3))

# Fonction pour prédire les tags d'une image avec un seuil de probabilité
def predict_image_tags(image_path, threshold=0.5):
    width, height = model.input_shape[2], model.input_shape[1]
    image = load_image_for_deepdanbooru(image_path, width, height)

    # Faire une prédiction en supprimant les logs inutiles
    predictions = model.predict(image, verbose=0)[0]

    # Obtenir les tags prédits avec des probabilités supérieures au seuil
    result_tags = []
    for i, score in enumerate(predictions):
        if score >= threshold:
            result_tags.append((tags_dict[i], score))

    # Créer un ensemble de tags pour une vérification plus rapide
    predicted_tags_set = set(tag for tag, score in result_tags)

    return predicted_tags_set, result_tags

# Fonction pour traiter toutes les images dans un dossier et détecter les personnages
def process_images_and_detect_characters(folder_path, threshold=0.5, match_threshold=0.5):
    # Obtenir la liste des fichiers d'images dans le dossier
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, extension)))

    # Vérifier s'il y a des images dans le dossier
    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {folder_path}.")
        return

    # Dictionnaire pour stocker les résultats
    character_detections = {character: [] for character in characters}

    # Dictionnaire pour stocker le pool de tags (optionnel)
    tags_pool = {}

    # Supprimer et recréer les sous-dossiers pour chaque personnage
    for character in characters:
        character_folder = os.path.join(folder_path, character)
        if os.path.exists(character_folder):
            shutil.rmtree(character_folder)
            print(f"Le dossier '{character}' existait déjà et a été supprimé.")
        os.makedirs(character_folder)
        print(f"Le dossier '{character}' a été créé.")

    # Supprimer et recréer le dossier 'zdivers' pour les images sans correspondance
    zdivers_folder = os.path.join(folder_path, 'zdivers')
    if os.path.exists(zdivers_folder):
        shutil.rmtree(zdivers_folder)
        print("Le dossier 'zdivers' existait déjà et a été supprimé.")
    os.makedirs(zdivers_folder)
    print("Le dossier 'zdivers' a été créé.")

    # Traiter chaque image
    for image_path in image_paths:
        predicted_tags_set, result_tags = predict_image_tags(image_path, threshold=threshold)

        # Ajouter les tags au pool de tags (optionnel)
        for tag, score in result_tags:
            if tag in tags_pool:
                tags_pool[tag] += 1
            else:
                tags_pool[tag] = 1

        # Vérifier chaque personnage
        image_characters = []  # Liste des personnages détectés dans cette image
        for character, char_tags in characters.items():
            # Compter le nombre de caractéristiques présentes
            matching_tags = predicted_tags_set.intersection(char_tags)
            match_ratio = len(matching_tags) / len(char_tags)

            if match_ratio >= match_threshold:
                image_characters.append(character)
                character_detections[character].append(image_path)

        # Copier l'image dans les dossiers correspondants
        if image_characters:
            for character in image_characters:
                character_folder = os.path.join(folder_path, character)
                destination_path = os.path.join(character_folder, os.path.basename(image_path))

                # Éviter l'écrasement si le fichier existe déjà
                base_name, extension = os.path.splitext(os.path.basename(image_path))
                counter = 1
                while os.path.exists(destination_path):
                    destination_path = os.path.join(character_folder, f"{base_name}_{counter}{extension}")
                    counter += 1

                shutil.copy2(image_path, destination_path)

            # Afficher une ligne indiquant dans quels dossiers l'image a été classée
            if len(image_characters) > 1:
                print(f"L'image '{os.path.basename(image_path)}' a été classée dans les dossiers : {', '.join(image_characters)}")
            else:
                print(f"L'image '{os.path.basename(image_path)}' a été classée dans le dossier : {image_characters[0]}")
        else:
            # Copier l'image dans 'zdivers'
            destination_path = os.path.join(zdivers_folder, os.path.basename(image_path))

            # Éviter l'écrasement si le fichier existe déjà
            base_name, extension = os.path.splitext(os.path.basename(image_path))
            counter = 1
            while os.path.exists(destination_path):
                destination_path = os.path.join(zdivers_folder, f"{base_name}_{counter}{extension}")
                counter += 1

            shutil.copy2(image_path, destination_path)
            print(f"L'image '{os.path.basename(image_path)}' a été classée dans le dossier : zdivers")

    # Afficher le résumé (optionnel)
    # Si vous ne souhaitez pas afficher le résumé, vous pouvez commenter ou supprimer cette partie
    """
    print("\n--- Résumé ---")
    print(f"Nombre total d'images traitées : {len(image_paths)}")
    for character, images in character_detections.items():
        print(f"Nombre d'images où '{character}' a été détecté : {len(images)}")

    # Afficher le pool de tags collectés (top 20)
    sorted_tags_pool = sorted(tags_pool.items(), key=lambda x: x[1], reverse=True)
    print("\nPool de tags collectés (top 20) :")
    for tag, count in sorted_tags_pool[:20]:
        print(f"{tag}: {count} occurrences")
    """

# Chemin vers le dossier contenant les images
folder_path = 'images'  # Remplacez par le chemin vers votre dossier d'images

# Appeler la fonction pour traiter les images du dossier
process_images_and_detect_characters(folder_path, threshold=0.5, match_threshold=0.5)

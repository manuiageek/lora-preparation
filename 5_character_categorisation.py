import deepdanbooru as dd
import numpy as np
from PIL import Image
import os
import glob
import shutil  # Pour gérer les fichiers et dossiers
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Désactiver les messages de log TensorFlow
tf.get_logger().setLevel('ERROR')

# Spécifiez le chemin vers le projet DeepDanbooru
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

# Fonction pour charger une image
def load_image(image_path, width, height):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image, dtype=np.float32) / 255.0
    return image

# Fonction pour prétraiter un batch d'images en parallèle
def load_images_for_deepdanbooru(image_paths, width, height):
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(lambda p: load_image(p, width, height), image_paths))
    images = np.stack(images, axis=0)
    return images  # Retourne un tableau numpy de forme (batch_size, height, width, 3)

# Fonction pour prédire les tags d'un batch d'images avec un seuil de probabilité
def predict_image_tags_batch(image_paths, threshold=0.5):
    width, height = model.input_shape[2], model.input_shape[1]
    images = load_images_for_deepdanbooru(image_paths, width, height)

    # Faire une prédiction en supprimant les logs inutiles
    predictions = model.predict(images, verbose=0)  # Prédictions pour tout le batch

    batch_results = []
    for idx, preds in enumerate(predictions):
        result_tags = []
        for i, score in enumerate(preds):
            if score >= threshold:
                result_tags.append((tags_dict[i], score))
        predicted_tags_set = set(tag for tag, score in result_tags)
        batch_results.append((image_paths[idx], predicted_tags_set, result_tags))
    return batch_results  # Retourne une liste de tuples (image_path, predicted_tags_set, result_tags)

# Fonction pour traiter un sous-dossier
def process_subfolder(subfolder_path, destination_folder, threshold=0.5, match_threshold=0.5, batch_size=32):
    subfolder_name = os.path.basename(subfolder_path)
    print(f"\nTraitement du dossier : {subfolder_name}")

    # Obtenir la liste des fichiers d'images dans le sous-dossier
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(subfolder_path, extension)))

    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {subfolder_path}.")
        return

    # Créer les dossiers de destination s'ils n'existent pas
    character_folders = {}
    for character in characters:
        character_folder = os.path.join(destination_folder, character)
        if not os.path.exists(character_folder):
            os.makedirs(character_folder)
            print(f"Le dossier '{character}' a été créé.")

        character_folders[character] = character_folder

    # Créer le dossier 'zboy' s'il n'existe pas
    zboy_folder = os.path.join(destination_folder, 'zboy')
    if not os.path.exists(zboy_folder):
        os.makedirs(zboy_folder)
        print("Le dossier 'zboy' a été créé.")

    # Créer le dossier 'zmisc' s'il n'existe pas
    zmisc_folder = os.path.join(destination_folder, 'zmisc')
    if not os.path.exists(zmisc_folder):
        os.makedirs(zmisc_folder)
        print("Le dossier 'zmisc' a été créé.")

    # Supprimer les fichiers commençant par le nom du sous-dossier dans les dossiers de destination
    prefix = subfolder_name.split('_')[0] + '_'
    for folder in list(character_folders.values()) + [zboy_folder, zmisc_folder]:
        for file in os.listdir(folder):
            if file.startswith(prefix):
                file_path = os.path.join(folder, file)
                os.remove(file_path)
                print(f"Le fichier '{file}' a été supprimé du dossier '{os.path.basename(folder)}'.")

    # Traiter les images par batches
    num_images = len(image_paths)
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_results = predict_image_tags_batch(batch_image_paths, threshold=threshold)

        # Traiter les résultats du batch
        for image_path, predicted_tags_set, result_tags in batch_results:
            # Obtenir le chemin relatif de l'image par rapport au dossier racine
            relative_image_path = os.path.relpath(image_path, destination_folder)

            # Vérifier la présence des tags "boy" et "girl"
            has_boy = any('boy' in tag for tag in predicted_tags_set)
            has_girl = any('girl' in tag for tag in predicted_tags_set)

            # Si l'image contient "boy" et pas "girl", la déplacer dans 'zboy'
            if has_boy and not has_girl:
                # Copier l'image dans 'zboy' avec le nom original
                original_filename = os.path.basename(image_path)
                new_filename = f"{subfolder_name}_{original_filename}"
                destination_path = os.path.join(zboy_folder, new_filename)

                # Éviter l'écrasement si le fichier existe déjà
                base_name, extension = os.path.splitext(new_filename)
                counter = 1
                while os.path.exists(destination_path):
                    new_filename = f"{base_name}_{counter}{extension}"
                    destination_path = os.path.join(zboy_folder, new_filename)
                    counter += 1

                shutil.copy2(image_path, destination_path)
                print(f"L'image '{subfolder_name}/{original_filename}' contient 'boy' et a été classée dans le dossier : zboy")
                continue  # Passer à l'image suivante

            # Vérifier chaque personnage
            image_characters = []  # Liste des personnages détectés dans cette image
            for character, char_tags in characters.items():
                # Compter le nombre de caractéristiques présentes
                matching_tags = predicted_tags_set.intersection(char_tags)
                match_ratio = len(matching_tags) / len(char_tags)

                if match_ratio >= match_threshold:
                    image_characters.append(character)

            if image_characters:
                # Copier l'image dans les dossiers correspondants avec le nouveau nom
                for character in image_characters:
                    character_folder = character_folders[character]
                    original_filename = os.path.basename(image_path)
                    new_filename = f"{subfolder_name}_{original_filename}"
                    destination_path = os.path.join(character_folder, new_filename)

                    # Éviter l'écrasement si le fichier existe déjà
                    base_name, extension = os.path.splitext(new_filename)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_filename = f"{base_name}_{counter}{extension}"
                        destination_path = os.path.join(character_folder, new_filename)
                        counter += 1

                    shutil.copy2(image_path, destination_path)

                # Afficher une ligne indiquant dans quels dossiers l'image a été classée
                if len(image_characters) > 1:
                    print(f"L'image '{subfolder_name}/{original_filename}' a été classée dans les dossiers : {', '.join(image_characters)}")
                else:
                    print(f"L'image '{subfolder_name}/{original_filename}' a été classée dans le dossier : {image_characters[0]}")
            else:
                # Si l'image contient 'boy' et 'girl' mais ne correspond à aucun personnage féminin, la déplacer dans 'zmisc'
                if has_boy and has_girl:
                    # Copier l'image dans 'zmisc' avec le nouveau nom
                    original_filename = os.path.basename(image_path)
                    new_filename = f"{subfolder_name}_{original_filename}"
                    destination_path = os.path.join(zmisc_folder, new_filename)

                    # Éviter l'écrasement si le fichier existe déjà
                    base_name, extension = os.path.splitext(new_filename)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_filename = f"{base_name}_{counter}{extension}"
                        destination_path = os.path.join(zmisc_folder, new_filename)
                        counter += 1

                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{subfolder_name}/{original_filename}' contient 'boy' et 'girl', mais n'a correspondu à aucun personnage. Classée dans 'zmisc'.")
                else:
                    # Copier l'image dans 'zmisc' avec le nouveau nom
                    original_filename = os.path.basename(image_path)
                    new_filename = f"{subfolder_name}_{original_filename}"
                    destination_path = os.path.join(zmisc_folder, new_filename)

                    # Éviter l'écrasement si le fichier existe déjà
                    base_name, extension = os.path.splitext(new_filename)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_filename = f"{base_name}_{counter}{extension}"
                        destination_path = os.path.join(zmisc_folder, new_filename)
                        counter += 1

                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{subfolder_name}/{original_filename}' a été classée dans le dossier : zmisc")

    print(f"Le dossier '{subfolder_name}' a été traité.")

# Fonction principale pour traiter tous les sous-dossiers
def process_all_subfolders(root_folder, destination_folder, threshold=0.5, match_threshold=0.5, batch_size=32):
    # Obtenir la liste des sous-dossiers à traiter
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    if not subfolders:
        print(f"Aucun sous-dossier trouvé dans le dossier {root_folder}.")
        return

    for subfolder in subfolders:
        # Ignorer les dossiers de destination (personnages, 'zboy' et 'zmisc')
        if os.path.basename(subfolder) in list(characters.keys()) + ['zboy', 'zmisc']:
            continue
        process_subfolder(subfolder, destination_folder, threshold, match_threshold, batch_size)

# Chemin vers le dossier contenant les images
root_folder = r'T:\_SELECT\__DUMBBELL NANKILO MOTERU'  # Dossier racine contenant les sous-dossiers à traiter
destination_folder = root_folder  # Les dossiers de destination se trouvent dans le dossier racine

# Appeler la fonction pour traiter tous les sous-dossiers
process_all_subfolders(root_folder, destination_folder, threshold=0.4, match_threshold=0.5, batch_size=16)

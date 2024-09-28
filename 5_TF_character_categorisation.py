import deepdanbooru as dd
import numpy as np
from PIL import Image
import os
import glob
import shutil
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision
from concurrent.futures import ThreadPoolExecutor

# Définir le nombre de cœurs CPU à utiliser
NUM_CORES = 12

# Spécifiez le chemin vers le projet DeepDanbooru
project_path = './models/deepdanbooru'

# Charger le modèle sans le compiler pour laisser le choix du périphérique plus tard
model = dd.project.load_model_from_project(project_path, compile_model=False)

# Charger les tags associés
tags = dd.project.load_tags_from_project(project_path)

# Convertir la liste des tags en un dictionnaire pour un accès plus rapide
tags_dict = {i: tag for i, tag in enumerate(tags)}

# Dictionnaire des personnages avec leurs caractéristiques (tags)
characters = {
    'akame_agk': ['black_hair', 'long_hair', 'straight_hair', 'red_eyes', 'bangs'],
    'chelsea_agk': ['long_hair', 'pink_hair', 'wavy_hair', 'ahoge', 'hair_behind_ears', 'red_eyes', 'hair_ribbon', 'headphones', 'bangs', 'hair_over_shoulder'],
    'Esdeath_agk': ['long_hair', 'light_blue_hair', 'straight_hair', 'blue_eyes', 'hat', 'ahoge'],
    'kurome_agk': ['short_hair', 'black_hair', 'straight_hair', 'bangs', 'hair_between_eyes', 'black_eyes'],
    'leone_agk': ['blonde_hair', 'yellow_eyes', 'messy_hair', 'medium_hair', 'bangs', 'ahoge', 'hair_over_shoulders','side_locks'],
    'mine_agk': ['pink_hair', 'twin_tails', 'long_hair', 'hair_ribbon', 'hair_between_eyes', 'bangs', 'pink_eyes'],
    'seryu_agk': ['brown_hair', 'long_hair', 'ponytail', 'bangs', 'hair_behind_ears', 'brown_eyes', 'hair_stick'],
    'sheele_agk': ['purple_hair', 'long_hair', 'glasses', 'bangs', 'hair_between_eyes', 'hair_ornament', 'purple_eyes', 'straight_hair', 'side_bangs']   
}

# Fonction pour charger une image et la redimensionner (CPU)
def load_and_resize_image(image_path, width, height):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image, dtype=np.float16) / 255.0  # Utiliser float16 pour la précision mixte
    return image

# Fonction pour charger toutes les images d'un sous-dossier dans la RAM
def load_all_images_from_subfolder(image_paths, width, height):
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:  # Utiliser le nombre de cœurs défini
        images = list(executor.map(lambda p: load_and_resize_image(p, width, height), image_paths))
    return images  # Retourne une liste d'images redimensionnées

# Fonction pour prédire les tags d'un batch d'images avec un seuil de probabilité
def predict_image_tags_batch(images, image_paths, threshold=0.5, device_type='gpu'):
    images = np.stack(images, axis=0)  # Convertir en tableau numpy pour l'inférence
    
    device = '/GPU:0' if device_type == 'gpu' else '/CPU:0'
    try:
        # Essayer d'effectuer la prédiction sur le périphérique sélectionné
        with tf.device(device):
            predictions = model.predict(images, verbose=0)
    except tf.errors.ResourceExhaustedError:
        print("Mémoire GPU insuffisante, basculement vers le CPU.")
        # Effectuer la prédiction sur le CPU en cas d'erreur de mémoire GPU
        with tf.device('/CPU:0'):
            predictions = model.predict(images, verbose=0)

    # Convertir les prédictions en float32 pour éviter les problèmes de compatibilité
    predictions = predictions.astype(np.float32)

    batch_results = []
    for idx, preds in enumerate(predictions):
        result_tags = []
        for i, score in enumerate(preds):
            if score >= threshold:
                result_tags.append((tags_dict[i], score))
        predicted_tags_set = set(tag for tag, score in result_tags)
        batch_results.append((image_paths[idx], predicted_tags_set, result_tags))
    return batch_results

# Fonction pour nettoyer les fichiers précédemment classés
def clean_previous_classifications(destination_folder, subfolder_name):
    # Parcourir tous les dossiers de destination et supprimer les fichiers qui commencent par "subfolder_name_"
    folders_to_clean = [os.path.join(destination_folder, character) for character in characters] + \
                       [os.path.join(destination_folder, 'zboy'), os.path.join(destination_folder, 'zmisc')]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.startswith(f"{subfolder_name}_"):
                    file_path = os.path.join(folder, file)
                    os.remove(file_path)
                    print(f"Suppression du fichier existant : {file_path}")

# Fonction pour traiter un sous-dossier
def process_subfolder(subfolder_path, destination_folder, threshold=0.5, match_threshold=0.5, batch_size=16, device_type='gpu'):
    subfolder_name = os.path.basename(subfolder_path)
    print(f"\nTraitement du dossier : {subfolder_name}")

    # Nettoyer les fichiers déjà classés pour ce sous-dossier
    clean_previous_classifications(destination_folder, subfolder_name)

    # Obtenir la liste des fichiers d'images dans le sous-dossier
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(subfolder_path, extension)))

    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {subfolder_path}.")
        return

    # Charger toutes les images du sous-dossier en RAM
    width, height = model.input_shape[2], model.input_shape[1]
    images = load_all_images_from_subfolder(image_paths, width, height)

    # Créer les dossiers de destination s'ils n'existent pas
    character_folders = {}
    for character in characters:
        character_folder = os.path.join(destination_folder, character)
        if not os.path.exists(character_folder):
            os.makedirs(character_folder)
        character_folders[character] = character_folder

    zboy_folder = os.path.join(destination_folder, 'zboy')
    if not os.path.exists(zboy_folder):
        os.makedirs(zboy_folder)

    zmisc_folder = os.path.join(destination_folder, 'zmisc')
    if not os.path.exists(zmisc_folder):
        os.makedirs(zmisc_folder)

    # Traiter les images par lots
    num_images = len(image_paths)
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_results = predict_image_tags_batch(batch_images, batch_image_paths, threshold=threshold, device_type=device_type)

        # Traiter les résultats du batch
        for image_path, predicted_tags_set, result_tags in batch_results:
            has_boy = any('boy' in tag for tag in predicted_tags_set)
            has_girl = any('girl' in tag for tag in predicted_tags_set)

            if has_boy and not has_girl:
                # Nouveau nom de fichier avec le nom du sous-dossier
                new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                destination_path = os.path.join(zboy_folder, new_filename)
                shutil.copy2(image_path, destination_path)
                print(f"L'image '{image_path}' a été classée dans 'zboy'.")
            else:
                image_characters = []
                for character, char_tags in characters.items():
                    matching_tags = predicted_tags_set.intersection(char_tags)
                    match_ratio = len(matching_tags) / len(char_tags)

                    if match_ratio >= match_threshold:
                        image_characters.append(character)

                if image_characters:
                    for character in image_characters:
                        # Nouveau nom de fichier avec le nom du sous-dossier
                        new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                        destination_path = os.path.join(character_folders[character], new_filename)
                        shutil.copy2(image_path, destination_path)
                    print(f"L'image '{image_path}' a été classée dans : {', '.join(image_characters)}")
                else:
                    # Nouveau nom de fichier avec le nom du sous-dossier
                    new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                    destination_path = os.path.join(zmisc_folder, new_filename)
                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{image_path}' a été classée dans 'zmisc'.")

    print(f"Le dossier '{subfolder_name}' a été traité.")

# Fonction principale pour traiter tous les sous-dossiers
def process_all_subfolders(root_folder, destination_folder, threshold=0.4, match_threshold=0.5, batch_size=16, device_type='gpu'):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print(f"Aucun sous-dossier trouvé dans le dossier {root_folder}.")
        return

    for subfolder in subfolders:
        if os.path.basename(subfolder) in list(characters.keys()) + ['zboy', 'zmisc']:
            continue
        process_subfolder(subfolder, destination_folder, threshold, match_threshold, batch_size, device_type)

# Chemin vers le dossier contenant les images
root_folder = r'F:\3_TO_DETECT\_AKAME GA KILL'
destination_folder = root_folder

# Appeler la fonction pour traiter tous les sous-dossiers
if __name__ == '__main__':
    process_all_subfolders(root_folder, destination_folder, threshold=0.4, match_threshold=0.5, batch_size=16, device_type='cpu')
    print(f"Traitement terminé. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

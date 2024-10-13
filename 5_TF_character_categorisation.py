import deepdanbooru as dd
import numpy as np
from PIL import Image
import os
import glob
import shutil
from datetime import datetime
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import psutil

# Chemin vers le dossier contenant les images
root_folder = r'T:\_SELECT\X_-DUMBBELL NAN KILO MOTERU'

# Dictionnaire des personnages avec leurs caractéristiques (tags)
characters = {
'akemi_dnm': ['bangs', 'black_hair', 'blunt_bangs', 'long_hair','aqua_eyes'],
'ayaka_dnm': ['blue_sky', 'blur_censor', 'bokeh', 'brown_hair', 'hair_bun', 'hair_ornament','bun', 'asymmetrical_bangs', 'tan_skin', 'purple_eyes'],     
'crystal_dnm': ['blue_eyes', 'blur_censor', 'long_hair','white_hair'],
'gina_dnm': ['bangs', 'black_serafuku', 'blue_eyes', 'blunt_bangs','short_hair','white_hair', 'bob_cut'], 
'hibiki_dnm': ['bangs', 'black_tank_top', 'blonde_hair', 'green_eyes', 'gyaru', 'long_hair', 'twintails'], 
'satomi_dnm': ['brown_eyes','short_hair', 'black_hair', 'bob_cut'],
}

# Configuration centrale des paramètres
device_type = 'gpu'  # 'gpu' ou 'cpu' selon vos besoins
NUM_CORES = 24  # 8, 24 ou 32 cœurs CPU à utiliser
BATCH_SIZE = 16  # Taille du batch pour le traitement des images
MAX_MEMORY_BYTES = 10 * 1024 ** 3  # Limite de RAM allouée en bytes (10 Goctets)

# Force le CPU si nécessaire
if device_type == 'cpu':
    tf.config.set_visible_devices([], 'GPU')

# Définir l'affinité des cœurs CPU pour le script TensorFlow
p = psutil.Process()  # Obtenir le processus actuel

if NUM_CORES == 8:
    # Pour 8 cœurs physiques + HT
    p.cpu_affinity([0, 1, 2, 3, 16, 17, 18, 19])
elif NUM_CORES == 24:
    # Pour 24 cœurs physiques + HT (12 cœurs physiques + 12 HT)
    p.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
elif NUM_CORES == 32:
    # Pour 32 cœurs physiques + HT (16 cœurs physiques + 16 HT)
    p.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
else:
    # Affinité par défaut pour tous les cœurs disponibles
    available_cores = list(range(psutil.cpu_count()))
    p.cpu_affinity(available_cores)

# Spécifiez le chemin vers le projet DeepDanbooru
project_path = './models/deepdanbooru'

# Charger le modèle sans le compiler pour laisser le choix du périphérique plus tard
model = dd.project.load_model_from_project(project_path, compile_model=False)

# Charger les tags associés
tags = dd.project.load_tags_from_project(project_path)

# Convertir la liste des tags en un dictionnaire pour un accès plus rapide
tags_dict = {i: tag for i, tag in enumerate(tags)}

# Fonction pour charger une image et la redimensionner (CPU)
def load_and_resize_image(image_path, width, height):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image, dtype=np.float16) / 255.0  # Utiliser float16 pour la précision mixte
    return image

# Fonction pour détecter si une image est de nuit en fonction de sa luminosité
def is_night_image(image_path, dark_threshold=50, dark_pixel_ratio=0.6):
    image = Image.open(image_path).convert('L')  # Convertir l'image en niveaux de gris
    pixels = np.array(image)  # Convertir l'image en un tableau numpy
    num_dark_pixels = np.sum(pixels < dark_threshold)  # Compter les pixels sombres
    total_pixels = pixels.size  # Nombre total de pixels

    # Calculer la proportion de pixels sombres
    dark_ratio = num_dark_pixels / total_pixels
    return dark_ratio >= dark_pixel_ratio  # Retourne True si l'image est considérée comme de nuit

# Fonction pour charger les images d'un batch
def load_images_batch(image_paths, width, height):
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        images = list(executor.map(lambda p: load_and_resize_image(p, width, height), image_paths))
    return images

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
        with tf.device('/CPU:0'):
            predictions = model.predict(images, verbose=0)

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
def clean_previous_classifications(root_folder, subfolder_name):
    folders_to_clean = [os.path.join(root_folder, character) for character in characters] + \
                       [os.path.join(root_folder, 'zboy'),
                        os.path.join(root_folder, 'z_daymisc'),
                        os.path.join(root_folder, 'z_nightmisc'),
                        os.path.join(root_folder, 'z_daymisc_girl'),
                        os.path.join(root_folder, 'z_nightmisc_girl'),
                        os.path.join(root_folder, 'z_background')]

    for folder in folders_to_clean:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.startswith(f"{subfolder_name}_"):
                    file_path = os.path.join(folder, file)
                    os.remove(file_path)
                    print(f"Suppression du fichier existant : {file_path}")

# Fonction pour traiter un sous-dossier
def process_subfolder(subfolder_path, root_folder, threshold=0.5, match_threshold=0.5, batch_size=BATCH_SIZE, device_type='gpu'):
    subfolder_name = os.path.basename(subfolder_path)
    print(f"\nTraitement du dossier : {subfolder_name}")

    # Nettoyer les fichiers déjà classés pour ce sous-dossier
    clean_previous_classifications(root_folder, subfolder_name)

    # Obtenir la liste des fichiers d'images dans le sous-dossier
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(subfolder_path, extension)))

    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {subfolder_name}.")
        return

    # Trier les chemins d'images dans l'ordre croissant
    image_paths.sort()

    # Créer les dossiers de destination s'ils n'existent pas
    character_folders = {}
    for character in characters:
        character_folder = os.path.join(root_folder, character)
        if not os.path.exists(character_folder):
            os.makedirs(character_folder)
        character_folders[character] = character_folder

    zboy_folder = os.path.join(root_folder, 'zboy')
    if not os.path.exists(zboy_folder):
        os.makedirs(zboy_folder)

    # Dossiers pour les images de filles sans personnage spécifique
    z_daymisc_girl_folder = os.path.join(root_folder, 'z_daymisc_girl')
    if not os.path.exists(z_daymisc_girl_folder):
        os.makedirs(z_daymisc_girl_folder)

    z_nightmisc_girl_folder = os.path.join(root_folder, 'z_nightmisc_girl')
    if not os.path.exists(z_nightmisc_girl_folder):
        os.makedirs(z_nightmisc_girl_folder)

    # Dossiers pour les images sans filles ni garçons mais avec personnes
    z_daymisc_folder = os.path.join(root_folder, 'z_daymisc')
    if not os.path.exists(z_daymisc_folder):
        os.makedirs(z_daymisc_folder)

    z_nightmisc_folder = os.path.join(root_folder, 'z_nightmisc')
    if not os.path.exists(z_nightmisc_folder):
        os.makedirs(z_nightmisc_folder)

    # Dossier pour les images sans aucune personne
    z_background_folder = os.path.join(root_folder, 'z_background')
    if not os.path.exists(z_background_folder):
        os.makedirs(z_background_folder)

    # Définir les ensembles de tags pour détecter les filles, les garçons et les personnes
    girl_tags = {'1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls', 'tomboy', 'demon_girl', 'fox_girl', 'fish_girl', 'arthropod_girl', 'lion_girl', 'tiger_girl', 'lamia_girl', 'old_woman', 'policewoman', 'woman'}
    boy_tags = {'1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys', 'fat_man', 'old_man', 'salaryman', 'ugly_man', 'man'}
    person_tags = girl_tags.union(boy_tags).union({'solo', 'multiple_persons', 'group'})

    # Obtenir la taille d'entrée du modèle
    width, height = model.input_shape[2], model.input_shape[1]

    # Calculer la taille approximative par image
    per_image_size = width * height * 3 * 2  # float16, 3 canaux, 2 bytes par valeur

    # Calculer le nombre maximal d'images pouvant être chargées en mémoire
    max_images_per_chunk = int(MAX_MEMORY_BYTES / per_image_size)
    total_images = len(image_paths)
    total_chunks = (total_images + max_images_per_chunk - 1) // max_images_per_chunk

    print(f"Nombre total d'images : {total_images}")
    print(f"Nombre maximal d'images par lot : {max_images_per_chunk}")
    print(f"Nombre total de lots : {total_chunks}")

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * max_images_per_chunk
        end_idx = min(start_idx + max_images_per_chunk, total_images)
        chunk_image_paths = image_paths[start_idx:end_idx]

        print(f"\nChargement du lot {chunk_idx + 1}/{total_chunks}, images {start_idx + 1} à {end_idx}")

        # Charger les images du lot en mémoire
        images_cache = {}
        with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
            loaded_images = list(executor.map(lambda p: (p, load_and_resize_image(p, width, height)), chunk_image_paths))

        # Trier le cache des images dans l'ordre croissant
        loaded_images.sort(key=lambda x: x[0])

        # Stocker les images dans le cache
        for img_path, img_array in loaded_images:
            images_cache[img_path] = img_array

        # Traiter les images par lots
        num_images_in_chunk = len(chunk_image_paths)
        for batch_start_idx in range(0, num_images_in_chunk, batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, num_images_in_chunk)
            batch_image_paths = chunk_image_paths[batch_start_idx:batch_end_idx]
            batch_images = [images_cache[p] for p in batch_image_paths]
            batch_results = predict_image_tags_batch(batch_images, batch_image_paths, threshold=threshold, device_type=device_type)

            # Traiter les résultats du batch
            for image_path, predicted_tags_set, result_tags in batch_results:
                # Afficher l'image en cours de traitement
                print(f"Traitement de l'image : {image_path}")

                # D'abord, vérifier si l'image est de nuit
                is_night = is_night_image(image_path)

                # Si l'image est de nuit
                if is_night:
                    # Vérifier si une fille est présente
                    has_girl = any(tag in predicted_tags_set for tag in girl_tags)
                    new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                    if has_girl:
                        destination_path = os.path.join(z_nightmisc_girl_folder, new_filename)
                        shutil.copy2(image_path, destination_path)
                        print(f"L'image '{image_path}' a été classée dans 'z_nightmisc_girl'")
                    else:
                        destination_path = os.path.join(z_nightmisc_folder, new_filename)
                        shutil.copy2(image_path, destination_path)
                        print(f"L'image '{image_path}' a été classée dans 'z_nightmisc'")
                    continue  # L'image a été classée, on passe à l'image suivante

                # Ensuite, si l'image n'est pas de nuit, vérifier les personnages et les autres tags
                has_girl = any(tag in predicted_tags_set for tag in girl_tags)
                has_boy = any(tag in predicted_tags_set for tag in boy_tags)
                has_person = any(tag in predicted_tags_set for tag in person_tags)

                if has_girl:
                    # L'image contient une fille, procéder à la classification des personnages
                    image_characters = []
                    for character, char_tags in characters.items():
                        matching_tags = predicted_tags_set.intersection(char_tags)
                        match_ratio = len(matching_tags) / len(char_tags)

                        if match_ratio >= match_threshold:
                            image_characters.append(character)

                    if image_characters:
                        # L'image correspond à un ou plusieurs personnages
                        for character in image_characters:
                            new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                            destination_path = os.path.join(character_folders[character], new_filename)
                            shutil.copy2(image_path, destination_path)
                        print(f"L'image '{image_path}' a été classée dans : {', '.join(image_characters)}")
                    else:
                        # Si l'image ne correspond à aucun personnage, classer jour/nuit
                        new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                        destination_path = os.path.join(z_daymisc_girl_folder, new_filename)
                        shutil.copy2(image_path, destination_path)
                        print(f"L'image '{image_path}' a été classée dans 'z_daymisc_girl'")
                elif has_boy:
                    # L'image contient un garçon (et pas de fille), la classer dans 'zboy'
                    new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                    destination_path = os.path.join(zboy_folder, new_filename)
                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{image_path}' a été classée dans 'zboy'")
                elif has_person:
                    # L'image contient une personne (ni garçon, ni fille spécifique), classer dans misc
                    new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                    destination_path = os.path.join(z_daymisc_folder, new_filename)
                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{image_path}' a été classée dans 'z_daymisc'")
                else:
                    # L'image ne contient aucune personne, la classer dans 'z_background'
                    new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                    destination_path = os.path.join(z_background_folder, new_filename)
                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{image_path}' a été classée dans 'z_background'")

        # Libérer la mémoire du cache
        images_cache.clear()
        print(f"Lot {chunk_idx + 1}/{total_chunks} traité et mémoire libérée.")

    print(f"Le dossier '{subfolder_name}' a été traité")

# Fonction principale pour traiter tous les sous-dossiers
def process_all_subfolders(root_folder, threshold=0.5, match_threshold=0.5, batch_size=BATCH_SIZE, device_type='gpu'):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print(f"Aucun sous-dossier trouvé dans le dossier {root_folder}")
        return

    for subfolder in subfolders:
        if os.path.basename(subfolder) in list(characters.keys()) + ['zboy', 'z_daymisc', 'z_nightmisc', 'z_daymisc_girl', 'z_nightmisc_girl', 'z_background']:
            continue
        process_subfolder(subfolder, root_folder, threshold, match_threshold, batch_size, device_type)

# Appeler la fonction pour traiter tous les sous-dossiers
if __name__ == '__main__':
    process_all_subfolders(root_folder, threshold=0.5, match_threshold=0.5, batch_size=BATCH_SIZE, device_type=device_type)
    print(f"Traitement terminé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

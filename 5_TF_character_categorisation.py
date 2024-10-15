import deepdanbooru as dd
import numpy as np
import os
import glob
import shutil
from datetime import datetime
import tensorflow as tf
import psutil
from concurrent.futures import ThreadPoolExecutor
import gc
from tensorflow.keras import mixed_precision

# Chemin vers le dossier contenant les images
root_folder = r'T:\_SELECT\X_-GRANCREST SENKI'

# Dictionnaire des personnages avec leurs caractéristiques (tags)
characters = {
'aishela_gcs': ['long_hair', 'red_eyes','purple_hair'],
'emaluna_gcs': ['green_eyes', 'long_hair','side_ponytail','silver_hair','bangs'],
'laura_gcs': ['blonde_hair', 'blue_eyes', 'hair_bun', 'long_hair', 'parted_hair','parted_bangs'],
'margret_gcs': ['hair_between_eyes', 'long_hair', 'red_eyes', 'red_hair','bangs'],
'marrine_gcs': ['blonde_hair', 'blue_eyes', 'long_hair', 'parted_hair','parted_bangs'],
'priscilla_gcs': ['green_eyes', 'long_hair', 'pink_hair','hair_between_eyes','bangs'],
'siluca_gcs': ['blonde_hair', 'long_hair', 'purple_eyes','hair_between_eyes'],
}

# Définir les constantes pour le seuil de prédiction et le seuil de correspondance
THRESHOLD = 0.35
MATCH_THRESHOLD = 0.65

# Configuration centrale des paramètres
device_type = 'gpu'  # 'gpu' ou 'cpu' selon vos besoins
NUM_CORES = 16  # Nombre de cœurs CPU à utiliser
BATCH_SIZE = 16  # Taille du batch pour le traitement des images
MAX_MEMORY_BYTES = 12 * 1024 ** 3  # Limite de RAM allouée en bytes (12 Goctets)

# Activer la précision mixte
mixed_precision.set_global_policy('mixed_float16')

# Configurer la croissance de la mémoire GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)

# Force le CPU si nécessaire
if device_type == 'cpu':
    tf.config.set_visible_devices([], 'GPU')

# Définir l'affinité des cœurs CPU pour le script TensorFlow
p = psutil.Process()  # Obtenir le processus actuel

if NUM_CORES == 8:
    # Utilisation des cœurs physiques 0 à 3, et SMT 17 à 19 (en évitant 16)
    p.cpu_affinity([0, 1, 2, 3, 17, 18, 19, 20])
elif NUM_CORES == 16:
    # Utiliser les 8 cœurs physiques (0 à 7) et leurs SMT (16 à 23)
    p.cpu_affinity([i for i in range(8)] + [i + 16 for i in range(8)])
elif NUM_CORES == 24:
    # Utiliser les 12 cœurs physiques (0 à 11) et leurs SMT (16 à 27)
    p.cpu_affinity([i for i in range(12)] + [i + 16 for i in range(12)])
else:
    # Si aucun des cas ne correspond, utiliser tous les cœurs disponibles
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

# Fonction pour détecter si une image est de nuit en utilisant TensorFlow
def is_night_image_from_tensor(image_tensor, dark_threshold=0.2, dark_pixel_ratio=0.6):
    image_gray = tf.reduce_mean(image_tensor, axis=2)
    num_dark_pixels = tf.reduce_sum(tf.cast(image_gray < dark_threshold, tf.float32))
    total_pixels = tf.cast(tf.size(image_gray), tf.float32)
    dark_ratio = num_dark_pixels / total_pixels
    return dark_ratio >= dark_pixel_ratio

# Fonction pour prédire les tags d'un batch d'images avec un seuil de probabilité
def predict_image_tags_batch(images, image_paths, threshold=THRESHOLD):
    device = '/GPU:0' if device_type == 'gpu' else '/CPU:0'
    try:
        with tf.device(device):
            predictions = model(images, training=False)
    except tf.errors.ResourceExhaustedError:
        print("Mémoire GPU insuffisante, basculement vers le CPU.")
        with tf.device('/CPU:0'):
            predictions = model(images, training=False)

    predictions = tf.cast(predictions, tf.float32).numpy()

    batch_results = []
    for idx in range(predictions.shape[0]):
        preds = predictions[idx]
        result_tags = []
        for i, score in enumerate(preds):
            if score >= threshold:
                result_tags.append((tags_dict[i], score))
        predicted_tags_set = set(tag for tag, score in result_tags)
        image_path_str = image_paths[idx].decode('utf-8')
        batch_results.append((image_path_str, predicted_tags_set, result_tags))
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
def process_subfolder(subfolder_path, root_folder, threshold=THRESHOLD, match_threshold=MATCH_THRESHOLD, batch_size=BATCH_SIZE, device_type='gpu'):
    subfolder_name = os.path.basename(subfolder_path)
    print(f"\nTraitement du dossier : {subfolder_name}")

    clean_previous_classifications(root_folder, subfolder_name)

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(subfolder_path, extension)))

    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {subfolder_name}.")
        return

    image_paths.sort()

    character_folders = {}
    for character in characters:
        character_folder = os.path.join(root_folder, character)
        if not os.path.exists(character_folder):
            os.makedirs(character_folder)
        character_folders[character] = character_folder

    zboy_folder = os.path.join(root_folder, 'zboy')
    if not os.path.exists(zboy_folder):
        os.makedirs(zboy_folder)

    z_daymisc_girl_folder = os.path.join(root_folder, 'z_daymisc_girl')
    if not os.path.exists(z_daymisc_girl_folder):
        os.makedirs(z_daymisc_girl_folder)

    z_nightmisc_girl_folder = os.path.join(root_folder, 'z_nightmisc_girl')
    if not os.path.exists(z_nightmisc_girl_folder):
        os.makedirs(z_nightmisc_girl_folder)

    z_daymisc_folder = os.path.join(root_folder, 'z_daymisc')
    if not os.path.exists(z_daymisc_folder):
        os.makedirs(z_daymisc_folder)

    z_nightmisc_folder = os.path.join(root_folder, 'z_nightmisc')
    if not os.path.exists(z_nightmisc_folder):
        os.makedirs(z_nightmisc_folder)

    z_background_folder = os.path.join(root_folder, 'z_background')
    if not os.path.exists(z_background_folder):
        os.makedirs(z_background_folder)

    girl_tags = {'1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls', 'tomboy', 'demon_girl',
                 'fox_girl', 'fish_girl', 'arthropod_girl', 'lion_girl', 'tiger_girl', 'lamia_girl', 'old_woman',
                 'policewoman', 'woman'}
    boy_tags = {'1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys', 'fat_man', 'old_man',
                'salaryman', 'ugly_man', 'man'}
    person_tags = girl_tags.union(boy_tags).union({'solo', 'multiple_persons', 'group'})

    width, height = model.input_shape[2], model.input_shape[1]

    per_image_size = width * height * 3 * 2  # float16, 3 canaux, 2 bytes par valeur

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

        dataset = tf.data.Dataset.from_tensor_slices(chunk_image_paths)

        def load_and_preprocess_image_with_path(image_path):
            image = tf.io.read_file(image_path)
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])  # Définir explicitement la forme
            image = tf.image.convert_image_dtype(image, tf.float16)
            image = tf.image.resize(image, [height, width])
            return image_path, image

        dataset = dataset.map(load_and_preprocess_image_with_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        with ThreadPoolExecutor(max_workers=NUM_CORES) as copy_executor:
            for batch in dataset:
                image_paths_batch, images_batch = batch
                image_paths_batch = image_paths_batch.numpy()

                batch_results = predict_image_tags_batch(images_batch, image_paths_batch, threshold=threshold)

                copy_tasks = []

                for idx, (image_path_str, predicted_tags_set, result_tags) in enumerate(batch_results):
                    print(f"Traitement de l'image : {image_path_str}")

                    image_tensor = images_batch[idx]
                    is_night = is_night_image_from_tensor(image_tensor)

                    is_night = bool(is_night.numpy())

                    new_filename = f"{subfolder_name}_{os.path.basename(image_path_str)}"

                    if is_night:
                        has_girl = any(tag in predicted_tags_set for tag in girl_tags)
                        if has_girl:
                            destination_path = os.path.join(z_nightmisc_girl_folder, new_filename)
                            copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                            print(f"L'image '{image_path_str}' a été classée dans 'z_nightmisc_girl'")
                        else:
                            destination_path = os.path.join(z_nightmisc_folder, new_filename)
                            copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                            print(f"L'image '{image_path_str}' a été classée dans 'z_nightmisc'")
                        continue

                    has_girl = any(tag in predicted_tags_set for tag in girl_tags)
                    has_boy = any(tag in predicted_tags_set for tag in boy_tags)
                    has_person = any(tag in predicted_tags_set for tag in person_tags)

                    if has_girl:
                        image_characters = []
                        for character, char_tags in characters.items():
                            matching_tags = predicted_tags_set.intersection(char_tags)
                            match_ratio = len(matching_tags) / len(char_tags)

                            if match_ratio >= match_threshold:
                                image_characters.append(character)

                        if image_characters:
                            for character in image_characters:
                                destination_path = os.path.join(character_folders[character], new_filename)
                                copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                            print(f"L'image '{image_path_str}' a été classée dans : {', '.join(image_characters)}")
                        else:
                            destination_path = os.path.join(z_daymisc_girl_folder, new_filename)
                            copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                            print(f"L'image '{image_path_str}' a été classée dans 'z_daymisc_girl'")
                    elif has_boy:
                        destination_path = os.path.join(zboy_folder, new_filename)
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                        print(f"L'image '{image_path_str}' a été classée dans 'zboy'")
                    elif has_person:
                        destination_path = os.path.join(z_daymisc_folder, new_filename)
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                        print(f"L'image '{image_path_str}' a été classée dans 'z_daymisc'")
                    else:
                        destination_path = os.path.join(z_background_folder, new_filename)
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                        print(f"L'image '{image_path_str}' a été classée dans 'z_background'")

                for task in copy_tasks:
                    task.result()

                del images_batch
                del image_paths_batch
                del batch_results
                gc.collect()

        print(f"Lot {chunk_idx + 1}/{total_chunks} traité.")

    print(f"Le dossier '{subfolder_name}' a été traité")

# Fonction principale pour traiter tous les sous-dossiers
def process_all_subfolders(root_folder, threshold=THRESHOLD, match_threshold=MATCH_THRESHOLD, batch_size=BATCH_SIZE, device_type='gpu'):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print(f"Aucun sous-dossier trouvé dans le dossier {root_folder}")
        return

    for subfolder in subfolders:
        if os.path.basename(subfolder) in list(characters.keys()) + ['zboy', 'z_daymisc', 'z_nightmisc',
                                                                     'z_daymisc_girl', 'z_nightmisc_girl',
                                                                     'z_background']:
            continue
        process_subfolder(subfolder, root_folder, threshold, match_threshold, batch_size, device_type)

# Appeler la fonction pour traiter tous les sous-dossiers
if __name__ == '__main__':
    process_all_subfolders(root_folder, threshold=THRESHOLD, match_threshold=MATCH_THRESHOLD, batch_size=BATCH_SIZE, device_type=device_type)
    print(f"Traitement terminé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import argparse
import deepdanbooru as dd
import numpy as np
import ast
import os
import glob
import shutil
from datetime import datetime
import tensorflow as tf
import psutil
from concurrent.futures import ThreadPoolExecutor
import gc
from tensorflow.keras import mixed_precision

def is_night_image_from_tensor(image_tensor, dark_threshold=0.2, dark_pixel_ratio=0.6):
    # Fonction pour détecter si une image est de nuit en utilisant TensorFlow
    image_gray = tf.reduce_mean(image_tensor, axis=2)
    num_dark_pixels = tf.reduce_sum(tf.cast(image_gray < dark_threshold, tf.float32))
    total_pixels = tf.cast(tf.size(image_gray), tf.float32)
    dark_ratio = num_dark_pixels / total_pixels
    return dark_ratio >= dark_pixel_ratio

def predict_image_tags_batch(images, image_paths, model, tags_dict, threshold, device_type):
    # Fonction pour prédire les tags d'un batch d'images avec un seuil de probabilité
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
        image_path_str = image_paths[idx]
        if isinstance(image_path_str, bytes):
            image_path_str = image_path_str.decode('utf-8')
        batch_results.append((image_path_str, predicted_tags_set, result_tags))
    return batch_results

def clean_previous_classifications(root_folder, subfolder_name, characters):
    # Fonction pour nettoyer les fichiers précédemment classés
    folders_to_clean = [os.path.join(root_folder, character) for character in characters] + \
                       [os.path.join(root_folder, 'zboy'),
                        os.path.join(root_folder, 'z_daymisc'),
                        os.path.join(root_folder, 'z_nightmisc'),
                        os.path.join(root_folder, 'z_daymisc_girl'),
                        os.path.join(root_folder, 'z_nightmisc_girl'),
                        os.path.join(root_folder, 'z_background'),
                        os.path.join(root_folder, 'zboy', '1person'),
                        os.path.join(root_folder, 'z_daymisc_girl', '1person'),
                        os.path.join(root_folder, 'z_nightmisc_girl', '1person'),
                        os.path.join(root_folder, 'z_closed_eyes'),
                        os.path.join(root_folder, 'z_noface'),
                        os.path.join(root_folder, 'z_zoomface')]
    for folder in folders_to_clean:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.startswith(f"{subfolder_name}_"):
                    file_path = os.path.join(folder, file)
                    os.remove(file_path)
                    print(f"Suppression du fichier existant : {file_path}")

def process_subfolder(subfolder_path, root_folder, characters, model, tags_dict, params):
    # Fonction pour traiter un sous-dossier
    subfolder_name = os.path.basename(subfolder_path)
    print(f"\nTraitement du dossier : {subfolder_name}")

    clean_previous_classifications(root_folder, subfolder_name, characters)

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    # Utilisation de glob récursif pour récupérer les images dans tous les sous-dossiers
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(subfolder_path, "**", extension), recursive=True))
    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {subfolder_name}.")
        return
    image_paths.sort()

    # Création des dossiers par personnage
    character_folders = {}
    for character in characters:
        character_folder = os.path.join(root_folder, character)
        if not os.path.exists(character_folder):
            os.makedirs(character_folder)
        character_folders[character] = character_folder

    # Création des dossiers globaux et de classement
    zboy_folder = os.path.join(root_folder, 'zboy')
    os.makedirs(zboy_folder, exist_ok=True)
    zboy_1person_folder = os.path.join(zboy_folder, '1person')
    os.makedirs(zboy_1person_folder, exist_ok=True)

    z_daymisc_girl_folder = os.path.join(root_folder, 'z_daymisc_girl')
    os.makedirs(z_daymisc_girl_folder, exist_ok=True)
    z_daymisc_girl_1person_folder = os.path.join(z_daymisc_girl_folder, '1person')
    os.makedirs(z_daymisc_girl_1person_folder, exist_ok=True)

    z_nightmisc_girl_folder = os.path.join(root_folder, 'z_nightmisc_girl')
    os.makedirs(z_nightmisc_girl_folder, exist_ok=True)
    z_nightmisc_girl_1person_folder = os.path.join(z_nightmisc_girl_folder, '1person')
    os.makedirs(z_nightmisc_girl_1person_folder, exist_ok=True)

    z_daymisc_folder = os.path.join(root_folder, 'z_daymisc')
    os.makedirs(z_daymisc_folder, exist_ok=True)

    z_nightmisc_folder = os.path.join(root_folder, 'z_nightmisc')
    os.makedirs(z_nightmisc_folder, exist_ok=True)

    z_background_folder = os.path.join(root_folder, 'z_background')
    os.makedirs(z_background_folder, exist_ok=True)

    # Dossier pour les yeux fermés
    z_closed_eyes_folder = os.path.join(root_folder, 'z_closed_eyes')
    os.makedirs(z_closed_eyes_folder, exist_ok=True)

    # Dossiers pour les cas liés au visage
    z_noface_folder = os.path.join(root_folder, 'z_noface')
    os.makedirs(z_noface_folder, exist_ok=True)
    z_zoomface_folder = os.path.join(root_folder, 'z_zoomface')
    os.makedirs(z_zoomface_folder, exist_ok=True)

    # Définition des tags par genre
    girl_tags = {
        '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls', 'tomboy', 'demon_girl',
        'fox_girl', 'fish_girl', 'arthropod_girl', 'lion_girl', 'tiger_girl', 'lamia_girl', 'old_woman',
        'policewoman', 'woman'
    }
    boy_tags = {
        '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys', 'fat_man', 'old_man',
        'salaryman', 'ugly_man', 'man'
    }
    person_tags = girl_tags.union(boy_tags).union({'solo', 'multiple_persons', 'group'})
    one_person_tags = {'solo'}

    width, height = model.input_shape[2], model.input_shape[1]
    per_image_size = width * height * 3 * 2  # float32, 3 canaux, 2 bytes par valeur
    max_images_per_chunk = int(params['MAX_MEMORY_BYTES'] / per_image_size)
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
            image.set_shape([None, None, 3])
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.LANCZOS3)
            return image_path, image

        dataset = dataset.map(load_and_preprocess_image_with_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(params['BATCH_SIZE'])
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        with ThreadPoolExecutor(max_workers=params['NUM_CORES']) as copy_executor:
            for batch in dataset:
                image_paths_batch, images_batch = batch
                image_paths_batch = image_paths_batch.numpy()
                batch_results = predict_image_tags_batch(
                    images_batch,
                    image_paths_batch,
                    model,
                    tags_dict,
                    threshold=params['THRESHOLD'],
                    device_type=params['device_type']
                )
                copy_tasks = []
                for idx, (image_path_str, predicted_tags_set, result_tags) in enumerate(batch_results):
                    print(f"Traitement de l'image : {image_path_str}")
                    image_tensor = images_batch[idx]
                    is_night = bool(is_night_image_from_tensor(image_tensor).numpy())
                    new_filename = f"{subfolder_name}_{os.path.basename(image_path_str)}"
                    has_one_person = 'solo' in predicted_tags_set

                    # ===================
                    # Cas des yeux fermés 
                    # ===================
                    if has_one_person and 'closed_eyes' in predicted_tags_set:
                        destination_path = os.path.join(z_closed_eyes_folder, new_filename)
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                        print(f"L'image '{image_path_str}' a été classée dans 'z_closed_eyes'")
                        continue

                    # ===================
                    # Cas extreme_closeup 
                    # ===================
                    if has_one_person and 'extreme_closeup' in predicted_tags_set:
                        destination_path = os.path.join(z_zoomface_folder, new_filename)
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                        print(f"L'image '{image_path_str}' a été classée dans 'z_zoomface'")
                        continue

                    # ===================
                    # Cas facelesss 
                    # ===================
                    if has_one_person and 'faceless' in predicted_tags_set:
                        destination_path = os.path.join(z_noface_folder, new_filename)
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                        print(f"L'image '{image_path_str}' a été classée dans 'z_noface'")
                        continue

                    # =======================
                    # TRI JOUR/NUIT + GENRE
                    # =======================
                    if is_night:
                        has_girl = any(tag in predicted_tags_set for tag in girl_tags)
                        if has_girl:
                            if has_one_person:
                                destination_path = os.path.join(z_nightmisc_girl_1person_folder, new_filename)
                                print(f"L'image '{image_path_str}' a été classée dans 'z_nightmisc_girl/1person'")
                            else:
                                destination_path = os.path.join(z_nightmisc_girl_folder, new_filename)
                                print(f"L'image '{image_path_str}' a été classée dans 'z_nightmisc_girl'")
                            copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
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
                            if len(char_tags) == 0:
                                continue
                            matching_tags = predicted_tags_set.intersection(char_tags)
                            match_ratio = len(matching_tags) / len(char_tags)
                            if match_ratio >= params['MATCH_THRESHOLD']:
                                image_characters.append(character)
                        if image_characters:
                            for character in image_characters:
                                destination_path = os.path.join(character_folders[character], new_filename)
                                copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                            print(f"L'image '{image_path_str}' a été classée dans : {', '.join(image_characters)}")
                        else:
                            if has_one_person:
                                destination_path = os.path.join(z_daymisc_girl_1person_folder, new_filename)
                                print(f"L'image '{image_path_str}' a été classée dans 'z_daymisc_girl/1person'")
                            else:
                                destination_path = os.path.join(z_daymisc_girl_folder, new_filename)
                                print(f"L'image '{image_path_str}' a été classée dans 'z_daymisc_girl'")
                            copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
                    elif has_boy:
                        if has_one_person:
                            destination_path = os.path.join(zboy_1person_folder, new_filename)
                            print(f"L'image '{image_path_str}' a été classée dans 'zboy/1person'")
                        else:
                            destination_path = os.path.join(zboy_folder, new_filename)
                            print(f"L'image '{image_path_str}' a été classée dans 'zboy'")
                        copy_tasks.append(copy_executor.submit(shutil.copy2, image_path_str, destination_path))
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
                del images_batch, image_paths_batch, batch_results
                gc.collect()
        print(f"Lot {chunk_idx + 1}/{total_chunks} traité.")
    print(f"Le dossier '{subfolder_name}' a été traité")

def process_all_subfolders(root_folder, characters, model, tags_dict, params):
    # Fonction principale pour traiter tous les sous-dossiers immédiats du dossier racine
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print(f"Aucun sous-dossier trouvé dans le dossier {root_folder}")
        return
    for subfolder in subfolders:
        if os.path.basename(subfolder) in list(characters.keys()) + [
            'zboy', 'z_daymisc', 'z_nightmisc',
            'z_daymisc_girl', 'z_nightmisc_girl', 'z_background',
            'z_closed_eyes', 'z_noface', 'z_zoomface'
        ]:
            continue
        process_subfolder(subfolder, root_folder, characters, model, tags_dict, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de classification d'images")
    parser.add_argument(
        '--root_folder',
        type=str,
        default=r'T:\_SELECT\_READY\FAIRY TAIL 100 YEARS QUEST',  # Chemin par défaut, à adapter si nécessaire
        help="Chemin vers le dossier contenant les images"
    )    
    parser.add_argument(
        '--character_file',
        type=str,
        default=r'chartags\FAIRY TAIL 100 YEARS QUEST.csv',  # Chemin par défaut, à adapter
        help="Chemin vers le fichier CSV des personnages"
    )
    args = parser.parse_args()

    # Paramètres du script
    params = {
        'THRESHOLD': 0.45,
        'MATCH_THRESHOLD': 0.6,
        'device_type': 'gpu',  # 'gpu' ou 'cpu' selon vos besoins
        'NUM_CORES': 24,       # Nombre de cœurs à utiliser
        'BATCH_SIZE': 30,      # Taille du batch pour le traitement
        'MAX_MEMORY_BYTES': 32 * 1024 ** 3  # 32 Go
    }

    # Configuration de la précision TensorFlow
    mixed_precision.set_global_policy('float32')

    # Configurer la croissance de la mémoire GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(e)
    # Forcer le CPU si nécessaire
    if params['device_type'] == 'cpu':
        tf.config.set_visible_devices([], 'GPU')

    # Définir l'affinité des cœurs CPU
    p = psutil.Process()
    if params['NUM_CORES'] == 8:
        p.cpu_affinity([0, 1, 2, 3, 17, 18, 19, 20])
    elif params['NUM_CORES'] == 16:
        p.cpu_affinity([i for i in range(8)] + [i + 16 for i in range(8)])
    elif params['NUM_CORES'] == 24:
        p.cpu_affinity([i for i in range(12)] + [i + 16 for i in range(12)])
    else:
        available_cores = list(range(psutil.cpu_count()))
        p.cpu_affinity(available_cores)

    # Chemin vers le dossier contenant les images
    root_folder = args.root_folder

    # Chargement du dictionnaire des personnages depuis le fichier CSV
    with open(args.character_file, 'r', encoding='utf-8') as file:
        data = file.read()
    characters = ast.literal_eval("{" + data + "}")

    # Spécifier le chemin du projet DeepDanbooru
    project_path = './models/deepdanbooru'

    # Charger le modèle sans compilation pour laisser choisir le périphérique ultérieurement
    model = dd.project.load_model_from_project(project_path, compile_model=False)

    # Charger la liste des tags du projet
    tags = dd.project.load_tags_from_project(project_path)

    # Convertir la liste des tags en dictionnaire pour un accès rapide
    tags_dict = {i: tag for i, tag in enumerate(tags)}

    # Appeler la fonction de traitement pour tous les sous-dossiers
    process_all_subfolders(root_folder, characters, model, tags_dict, params)

    print(f"Traitement terminé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
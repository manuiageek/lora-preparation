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
root_folder = r'T:\_SELECT\X_-AKAME GA KILL'

# Dictionnaire des personnages avec leurs caractéristiques (tags)
characters = {
'akame_agk': ['black_hair', 'long_hair', 'red_eyes', 'red_eyes',  'blunt_bangs', 'straight_hair'],
'chelsea_agk': ['brown_hair', 'hair_between_eyes', 'long_hair', 'pink_hair','very_long_hair','red_eyes', 'ahoge', 'hair_bow', 'side_bangs'],
'Esdeath_agk': ['long_hair', 'blue_hair', 'blue_eyes',  'straight_hair'],
'kurome_agk': ['black_eyes', 'black_hair', 'hair_between_eyes', 'long_hair','short_hair', 'blunt_bangs'],
'leone_agk': ['blonde_hair', 'green_eyes', 'blonde_hair', 'yellow_eyes', 'spiky_hair', 'long_hair', 'messy_hair', 'ahoge', 'blunt_bangs'],
'mine_agk': ['hair_between_eyes','long_hair', 'purple_eyes', 'purple_hair','pink_hair', 'pink_eyes', 'twintails', 'hair_bow', 'blunt_bangs'],
'seryu_agk': ['brown_eyes', 'brown_hair','long_hair', 'side_ponytail', 'ahoge', 'blunt_bangs'],
'sheele_agk': ['long_hair', 'purple_eyes', 'purple_hair', 'blunt_bangs', 'hair_ornament'],
}

# Configuration centrale des paramètres
device_type = 'cpu'  # 'gpu' ou 'cpu' selon vos besoins
NUM_CORES = 12  # 8 ou 12 cœurs CPU à utiliser
BATCH_SIZE = NUM_CORES  # Taille du batch pour le traitement des images
vram_limit = 6000  # Limite de mémoire GPU en méga-octets

# Force le CPU si nécessaire
if device_type == 'cpu':
    tf.config.set_visible_devices([], 'GPU')

# Définir l'affinité des cœurs CPU pour le script TensorFlow
p = psutil.Process()  # Obtenir le processus actuel

if NUM_CORES == 8:
    p.cpu_affinity([0, 1, 2, 3, 16, 17, 18, 19])
else:
    p.cpu_affinity([8, 9, 10, 11, 12, 13, 28, 29, 30, 31, 24, 25])

# Limiter la mémoire GPU avec TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and device_type == 'gpu':
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vram_limit)]
        )
    except RuntimeError as e:
        print(e)

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

    # Traiter les images par lots sans charger toutes les images en mémoire
    num_images = len(image_paths)
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_images = load_images_batch(batch_image_paths, width, height)
        batch_results = predict_image_tags_batch(batch_images, batch_image_paths, threshold=threshold, device_type=device_type)

        # Traiter les résultats du batch
        for image_path, predicted_tags_set, result_tags in batch_results:
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
                    print(f"L'image '{image_path}' a été classée dans 'z_nightmisc_girl'.")
                else:
                    destination_path = os.path.join(z_nightmisc_folder, new_filename)
                    shutil.copy2(image_path, destination_path)
                    print(f"L'image '{image_path}' a été classée dans 'z_nightmisc'.")
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
                    print(f"L'image '{image_path}' a été classée dans 'z_daymisc_girl'.")
                    shutil.copy2(image_path, destination_path)
            elif has_boy:
                # L'image contient un garçon (et pas de fille), la classer dans 'zboy'
                new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                destination_path = os.path.join(zboy_folder, new_filename)
                shutil.copy2(image_path, destination_path)
                print(f"L'image '{image_path}' a été classée dans 'zboy'.")
            elif has_person:
                # L'image contient une personne (ni garçon, ni fille spécifique), classer dans misc
                new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                destination_path = os.path.join(z_daymisc_folder, new_filename)
                shutil.copy2(image_path, destination_path)
                print(f"L'image '{image_path}' a été classée dans 'z_daymisc'.")
            else:
                # L'image ne contient aucune personne, la classer dans 'z_background'
                new_filename = f"{subfolder_name}_{os.path.basename(image_path)}"
                destination_path = os.path.join(z_background_folder, new_filename)
                shutil.copy2(image_path, destination_path)
                print(f"L'image '{image_path}' a été classée dans 'z_background'.")

    print(f"Le dossier '{subfolder_name}' a été traité.")

# Fonction principale pour traiter tous les sous-dossiers
def process_all_subfolders(root_folder, threshold=0.5, match_threshold=0.3, batch_size=BATCH_SIZE, device_type='gpu'):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print(f"Aucun sous-dossier trouvé dans le dossier {root_folder}.")
        return

    for subfolder in subfolders:
        if os.path.basename(subfolder) in list(characters.keys()) + ['zboy', 'z_daymisc', 'z_nightmisc', 'z_daymisc_girl', 'z_nightmisc_girl', 'z_background']:
            continue
        process_subfolder(subfolder, root_folder, threshold, match_threshold, batch_size, device_type)

# Appeler la fonction pour traiter tous les sous-dossiers
if __name__ == '__main__':
    process_all_subfolders(root_folder, threshold=0.5, match_threshold=0.3, batch_size=BATCH_SIZE, device_type=device_type)
    print(f"Traitement terminé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

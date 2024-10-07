import deepdanbooru as dd
import numpy as np
from PIL import Image
import glob
import os
import tensorflow as tf

# Désactiver les messages de log TensorFlow
tf.get_logger().setLevel('ERROR')

# Spécifiez le chemin vers le projet DeepDanbooru (le dossier où vous avez extrait le modèle pré-entraîné)
project_path = './models/deepdanbooru'

# Charger le modèle
model = dd.project.load_model_from_project(project_path, compile_model=False)

# Charger les tags associés
tags = dd.project.load_tags_from_project(project_path)

# Convertir la liste des tags en un dictionnaire pour un accès plus rapide
tags_dict = {i: tag for i, tag in enumerate(tags)}

# Liste des tags à exclure
excluded_tags = [
    'solo', '1girl', '1boy', 'rating:safe', 'rating:questionable', 'rating:explicit','3d',
    'armor', 'shoulder_armor', 'open_mouth','closed_mouth', 'smile', ':d', 'parted_lips',
    'headwear', 'helmet', 'hat', 'shirt',
    'looking_at_viewer', 'upper_body', 'portrait', 'blurry_background','gradient_background','depth_of_field','aiming_at_viewer','blurry_foreground',
    'simple_background', 'blurry','border',
]

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

    # Faire une prédiction sans affichage de la barre de progression
    predictions = model.predict(image, verbose=0)[0]

    # Obtenir les tags prédits avec des probabilités supérieures au seuil
    result_tags = [tags_dict[i] for i, score in enumerate(predictions) if score >= threshold]

    # Filtrer les tags qui ne commencent pas par "rating:" et qui ne sont pas dans la liste des tags exclus
    filtered_tags = [tag for tag in result_tags if tag not in excluded_tags]

    return filtered_tags

# Fonction pour traiter toutes les images dans un dossier
def process_images_in_folder(folder_path, threshold=0.5):
    # Obtenir la liste des fichiers d'images dans le dossier
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, extension)))

    # Vérifier s'il y a des images dans le dossier
    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {folder_path}.")
        return

    # Traiter chaque image et stocker les résultats
    for idx, image_path in enumerate(image_paths):
        # Retirer l'extension de l'image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        tags = predict_image_tags(image_path, threshold=threshold)
        tags_str = ", ".join([f"'{tag}'" for tag in tags])  # Ajouter des quotes autour de chaque tag

        # Affichage des tags pour chaque image
        print(f"'{image_name}': [{tags_str}],")

# Chemin vers le dossier contenant les images
folder_path = 'images'  # Remplacez par le chemin vers votre dossier d'images

# Appeler la fonction pour traiter les images du dossier
process_images_in_folder(folder_path, threshold=0.5)

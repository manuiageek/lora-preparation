import os
import deepdanbooru as dd
import numpy as np
from PIL import Image
import glob
import tensorflow as tf

# Définir une constante pour le chemin du projet DeepDanbooru
PROJECT_PATH = os.path.join('.', 'models', 'deepdanbooru')

# Ajuster le niveau de logging de TensorFlow pour afficher plus de messages
tf.get_logger().setLevel('ERROR')
# On utilise que le CPU 
tf.config.set_visible_devices([], 'GPU')

# Charger le modèle avec un bloc try-except
try:
    print("Chargement du modèle...")
    model = dd.project.load_model_from_project(PROJECT_PATH, compile_model=False)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Charger les tags associés avec un bloc try-except
try:
    tags = dd.project.load_tags_from_project(PROJECT_PATH)
except Exception as e:
    print(f"Erreur lors du chargement des tags : {e}")

# Vérifier si le modèle et les tags ont été chargés correctement avant de continuer
if 'model' not in locals() or 'tags' not in locals():
    print("Impossible de continuer sans le modèle ou les tags.")
    exit()

# Convertir la liste des tags en un dictionnaire pour un accès plus rapide
tags_dict = {i: tag for i, tag in enumerate(tags)}

# Liste des tags à exclure
excluded_tags = [
    'solo', '1girl', '1boy', 'rating:safe', 'rating:questionable', 'rating:explicit', '3d',
    'armor', 'shoulder_armor', 'open_mouth', 'closed_mouth', 'smile', ':d', 'parted_lips',
    'headwear', 'helmet', 'hat', 'shirt',
    'looking_at_viewer', 'upper_body', 'portrait', 'blurry_background', 'gradient_background', 'depth_of_field', 'aiming_at_viewer', 'blurry_foreground',
    'simple_background', 'blurry', 'border',
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

    # Filtrer les tags qui ne sont pas dans la liste des tags exclus
    filtered_tags = [tag for tag in result_tags if tag not in excluded_tags]

    return filtered_tags

# Fonction pour traiter toutes les images dans un dossier
def process_images_in_folder(folder_path, threshold=0.5):
    # Obtenir la liste des fichiers d'images dans le dossier
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.webp')
    image_paths = []
    for extension in image_extensions:
        found_images = glob.glob(os.path.join(folder_path, extension))
        image_paths.extend(found_images)

    # Trier les images par nom dans l'ordre croissant
    image_paths.sort()

    # Vérifier s'il y a des images dans le dossier
    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {folder_path}.")
        return

    # Traiter chaque image et stocker les résultats
    for idx, image_path in enumerate(image_paths):
        print(f"==== {image_path} ====")
        try:
            # Retirer l'extension de l'image
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            tags = predict_image_tags(image_path, threshold=threshold)
            tags_str = ", ".join([f"'{tag}'" for tag in tags])  # Ajouter des quotes autour de chaque tag

            # Affichage des tags pour chaque image
            print(f"'{image_name}': [{tags_str}],")
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path} : {e}")

# Chemin vers le dossier contenant les images
folder_path = 'images'  # Remplacez par le chemin vers votre dossier d'images

# Appeler la fonction pour traiter les images du dossier
process_images_in_folder(folder_path, threshold=0.4)

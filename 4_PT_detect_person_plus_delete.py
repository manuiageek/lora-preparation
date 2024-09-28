import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Déterminer le device (GPU ou CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilisation du device : {device}")

# Ajuster le nombre de threads CPU pour PyTorch si on est sur CPU
num_processes = 8
if device == 'cpu':
    torch.set_num_threads(num_processes)
    batch_size = 1
else:
    batch_size = 16

# Charger le modèle YOLOv8 animeface pré-entraîné sur le bon device
model_path = Path('models') / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))  # Charger le modèle
model.to(device)  # Envoyer le modèle sur le bon device (CPU ou GPU)

# Répertoire de base contenant les sous-dossiers
base_folder = r"/media/hleet_user/HDD-EXT/2_TO_EPURATE/HAZUREWAKU"

# Taille de l'image pour réduire l'utilisation de la VRAM
target_size = (640, 640)  # Redimensionner les images à 640x640

# Fonction pour charger une image et redimensionner
def load_and_resize_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.resize(img, target_size), image_path
    return None, image_path

# Fonction pour traiter un lot d'images
def process_batch(images_batch, paths_batch):
    if device == 'cuda':
        autocast_context = torch.cuda.amp.autocast()
    else:
        autocast_context = torch.no_grad()  # Sur CPU, utiliser torch.no_grad()

    with autocast_context:
        results = model(images_batch)  # Appeler directement l'inférence sur les images

    for i, result in enumerate(results):
        animeface_detected = False

        for detection in result.boxes.data:
            if int(detection[-1]) == 0:  # Classe 0 pour les visages d'anime
                animeface_detected = True
                break

        if not animeface_detected:
            print(f"Rien détecté dans : {paths_batch[i]}. Suppression ...")
            os.remove(paths_batch[i])
        else:
            print(f"Image OK : {paths_batch[i]}")

# Fonction pour traiter un sous-dossier complet en parallèle
def process_subfolder(subfolder, batch_size):
    images_in_memory = []
    paths_in_memory = []

    # Charger toutes les images du sous-dossier en RAM en parallèle
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for filename in os.listdir(subfolder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(subfolder, filename)
                futures.append(executor.submit(load_and_resize_image, image_path))
        
        for future in futures:
            img_resized, image_path = future.result()
            if img_resized is not None:
                images_in_memory.append(img_resized)
                paths_in_memory.append(image_path)

    # Traiter les images par lots
    images_batch = []
    paths_batch = []

    for img, image_path in zip(images_in_memory, paths_in_memory):
        images_batch.append(img)
        paths_batch.append(image_path)

        if len(images_batch) == batch_size:
            process_batch(images_batch, paths_batch)
            images_batch = []  # Réinitialiser le lot après traitement
            paths_batch = []

    # Traiter les images restantes dans le sous-dossier
    if images_batch:
        process_batch(images_batch, paths_batch)

# Parcourir uniquement les sous-dossiers de `base_folder` et traiter chaque sous-dossier séparément
for subdir in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subdir)
    if os.path.isdir(subfolder_path):
        print(f"Chargement et traitement du sous-dossier : {subfolder_path}")
        process_subfolder(subfolder_path, batch_size)
        print(f"Traitement du sous-dossier {subfolder_path} terminé.")

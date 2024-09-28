import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

# Déterminer le device (GPU ou CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilisation du device : {device}")

# Ajuster le nombre de threads CPU pour PyTorch si on est sur CPU
if device == 'cpu':
    torch.set_num_threads(8)
    # Taille du lot (nombre d'images traitées en parallèle)
    batch_size = 1
else:
    # Taille du lot (nombre d'images traitées en parallèle)
    batch_size = 16

# Charger le modèle YOLOv8 animeface pré-entraîné sur le bon device
model_path = Path('models') / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))  # Charger le modèle
model.to(device)  # Envoyer le modèle sur le bon device (CPU ou GPU)

# Répertoire de base contenant les sous-dossiers
base_folder = r"/media/hleet_user/HDD-EXT/2_TO_EPURATE/HAZUREWAKU"

# Taille de l'image pour réduire l'utilisation de la VRAM
target_size = (640, 640)  # Redimensionner les images à 640x640

# Fonction pour charger toutes les images d'un sous-dossier en RAM
def load_images_into_memory(subfolder):
    images_in_memory = []
    paths_in_memory = []
    
    # Charger les fichiers uniquement dans le sous-dossier courant
    for filename in os.listdir(subfolder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(subfolder, filename)
            
            # Charger l'image dans la RAM
            img = cv2.imread(image_path)
            if img is not None:
                # Redimensionner l'image pour réduire l'utilisation de la mémoire
                img_resized = cv2.resize(img, target_size)
                images_in_memory.append(img_resized)
                paths_in_memory.append(image_path)
    
    return images_in_memory, paths_in_memory

# Fonction pour traiter un lot d'images
def process_batch(images_batch, paths_batch):
    # Utilisation du mode de précision mixte (AMP) pour optimiser l'utilisation de la mémoire
    if device == 'cuda':
        autocast_context = torch.cuda.amp.autocast()
    else:
        autocast_context = torch.no_grad()  # Sur CPU, on utilise torch.no_grad() pour éviter les calculs inutiles

    with autocast_context:
        results = model(images_batch)  # Appeler directement l'inférence sur les images

    for i, result in enumerate(results):
        animeface_detected = False

        # Vérifier les détections
        for detection in result.boxes.data:
            if int(detection[-1]) == 0:  # Classe 0 pour les visages d'anime
                animeface_detected = True
                break

        # Si aucune animeface n'est détectée, supprimer l'image
        if not animeface_detected:
            print(f"Rien détecté dans : {paths_batch[i]}. Suppression ...")
            os.remove(paths_batch[i])
        else:
            print(f"Image OK : {paths_batch[i]}")

# Fonction pour traiter un sous-dossier complet
def process_subfolder(subfolder, batch_size):
    if device == 'cuda':
        # Charger toutes les images du sous-dossier en RAM
        images_in_memory, paths_in_memory = load_images_into_memory(subfolder)
        
        images_batch = []
        paths_batch = []

        # Parcourir les images chargées en mémoire et les traiter par lots
        for img, image_path in zip(images_in_memory, paths_in_memory):
            images_batch.append(img)
            paths_batch.append(image_path)
            
            # Si le lot est plein, traiter les images
            if len(images_batch) == batch_size:
                process_batch(images_batch, paths_batch)
                images_batch = []  # Réinitialiser le lot après traitement
                paths_batch = []

        # Traiter les images restantes dans le sous-dossier
        if images_batch:
            process_batch(images_batch, paths_batch)
    else:
        # Traitement image par image sans charger toutes les images en mémoire
        for filename in os.listdir(subfolder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(subfolder, filename)

                # Charger l'image
                img = cv2.imread(image_path)
                if img is not None:
                    # Redimensionner l'image pour réduire l'utilisation de la mémoire
                    img_resized = cv2.resize(img, target_size)
                    process_batch([img_resized], [image_path])

# Parcourir uniquement les sous-dossiers de `base_folder` et traiter chaque sous-dossier séparément
for subdir in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subdir)
    if os.path.isdir(subfolder_path):
        print(f"Chargement et traitement du sous-dossier : {subfolder_path}")
        process_subfolder(subfolder_path, batch_size)
        print(f"Traitement du sous-dossier {subfolder_path} terminé.")

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ajuster le nombre de threads CPU pour PyTorch
torch.set_num_threads(20)

# Charger le modèle YOLOv8 animeface pré-entraîné
model = YOLO('yolov8x6_animeface.pt')

# Répertoire de base contenant les fichiers
base_folder = r'T:\_SELECT\GRANCREST SENKI\25'

# Taille du lot (nombre d'images traitées en parallèle)
batch_size = 16

# Taille de l'image pour réduire l'utilisation de la VRAM
target_size = (640, 640)  # Redimensionner les images à 640x640

# Fonction pour charger toutes les images d'un sous-sous-dossier en RAM
def load_images_into_memory(subfolder):
    images_in_memory = []
    paths_in_memory = []
    
    # Charger les fichiers dans le dossier racine ET dans les sous-dossiers
    for root, dirs, files in os.walk(subfolder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, filename)
                
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
    # Utilisation du mode de précision mixte (AMP) pour optimiser l'utilisation de la mémoire du GPU
    with torch.amp.autocast('cuda'):
        results = model(images_batch, conf=0.25, verbose=False)

    for i, result in enumerate(results):
        animeface_detected = False

        # Vérifier les détections
        for detection in result.boxes.data:
            if int(detection[-1]) == 0:  # Classe 0 pour les visages d'anime
                animeface_detected = True
                break

        # Si aucune animeface n'est détectée, supprimer l'image
        if not animeface_detected:
            print(f"Rien dans : {paths_batch[i]}. Suppression ...")
            os.remove(paths_batch[i])
        else:
            print(f"Image OK : {paths_batch[i]}")

# Fonction pour traiter un répertoire complet
def process_folder(folder, batch_size):
    # Charger tout le dossier (y compris les sous-dossiers) en mémoire
    images_in_memory, paths_in_memory = load_images_into_memory(folder)
    
    images_batch = []
    paths_batch = []

    # Parcourir les images chargées en mémoire et les traiter par lots
    for i, (img, image_path) in enumerate(zip(images_in_memory, paths_in_memory)):
        images_batch.append(img)
        paths_batch.append(image_path)
        
        # Si le lot est plein, traiter les images
        if len(images_batch) == batch_size:
            process_batch(images_batch, paths_batch)
            images_batch = []  # Réinitialiser le lot après traitement
            paths_batch = []

    # Traiter les images restantes dans le dossier
    if images_batch:
        process_batch(images_batch, paths_batch)

# Traiter le répertoire de base et ses sous-dossiers (si existants)
print(f"Chargement et traitement du dossier : {base_folder}")
process_folder(base_folder, batch_size)
print(f"Traitement du dossier {base_folder} terminé.")

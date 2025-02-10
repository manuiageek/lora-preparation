import os
import cv2
import torch
import psutil
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

# Configuration centrale des paramètres
device_type = 'gpu'  # Définir 'cpu' ou 'gpu' selon vos besoins
num_processes = 16   # Nombre de cœurs CPU pour le chargement des images
batch_size_gpu = 10  # Taille de lot pour le GPU
batch_size_cpu = num_processes  # Taille de lot pour le CPU
pool_size = 1000  # Taille du pool d'images

# Déterminer le périphérique (GPU ou CPU)
device = 'cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu'
print(f"Utilisation du périphérique : {device}")

# Configuration de l'affinité des cœurs CPU pour le script GPU (facultatif, dépend de ta machine)
if device == 'cuda':
    p = psutil.Process()  # Obtenir le processus actuel
    if num_processes == 8:
        p.cpu_affinity([0, 1, 2, 3, 17, 18, 19, 20])
    elif num_processes == 16:
        p.cpu_affinity([i for i in range(8)] + [i + 16 for i in range(8)])
    elif num_processes == 24:
        p.cpu_affinity([i for i in range(12)] + [i + 16 for i in range(12)])
    else:
        available_cores = list(range(psutil.cpu_count()))
        p.cpu_affinity(available_cores)

# Ajuster le nombre de threads CPU pour PyTorch si on est sur CPU
if device == 'cpu':
    torch.set_num_threads(num_processes)
    batch_size = batch_size_cpu
else:
    batch_size = batch_size_gpu

# Charger le modèle YOLOv8 animeface pré-entraîné
model_path = Path('models') / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))  # Charger le modèle

# Taille de l'image pour réduire l'utilisation de la VRAM
target_size = (640, 640)  # Redimensionner les images à 640x640

# Fonction pour charger une image et la redimensionner (CPU)
def load_and_resize_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.resize(img, target_size), image_path
    return None, image_path

# Fonction pour traiter un lot d'images (GPU ou CPU)
def process_batch(images_batch, paths_batch, model, device_type='gpu'):
    try:
        device_local = 'cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu'
        model.to(device_local)  # Envoyer le modèle sur le périphérique sélectionné
        
        if device_local == 'cuda':
            autocast_context = torch.amp.autocast('cuda')
        else:
            autocast_context = torch.no_grad()  # Sur CPU, utiliser torch.no_grad()

        with autocast_context:
            # Désactiver les logs de prédiction (verbose=False)
            results = model(images_batch, verbose=False)

        for i, result in enumerate(results):
            animeface_detected = False

            for detection in result.boxes.data:
                # Classe 0 pour les visages d'anime (selon ton jeu de classes YOLO)
                if int(detection[-1]) == 0:
                    animeface_detected = True
                    break

            if not animeface_detected:
                print(f"Rien détecté dans : {paths_batch[i]}. Suppression ...")
                os.remove(paths_batch[i])
            else:
                print(f"Image OK : {paths_batch[i]}")

    except torch.cuda.OutOfMemoryError:
        print("Mémoire GPU insuffisante, basculement vers le CPU.")
        process_batch(images_batch, paths_batch, model, 'cpu')  # Retenter le traitement sur le CPU

# Générateur pour charger les images par lot
def image_generator(subfolder, pool_size):
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for filename in os.listdir(subfolder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(subfolder, filename)
                futures.append(executor.submit(load_and_resize_image, image_path))
                if len(futures) >= pool_size:
                    for future in futures:
                        img_resized, image_path = future.result()
                        if img_resized is not None:
                            yield img_resized, image_path
                    futures = []
        # Traiter les images restantes
        for future in futures:
            img_resized, image_path = future.result()
            if img_resized is not None:
                yield img_resized, image_path

# Fonction pour traiter un sous-dossier complet en parallèle (redimensionnement sur CPU)
def process_subfolder(subfolder, batch_size, model, device_type='gpu', num_processes=12, pool_size=1000):
    images_batch = []
    paths_batch = []
    
    gen = image_generator(subfolder, pool_size)
    
    for img, image_path in gen:
        images_batch.append(img)
        paths_batch.append(image_path)

        if len(images_batch) == batch_size:
            process_batch(images_batch, paths_batch, model, device_type)
            images_batch = []
            paths_batch = []
    
    # Traiter les images restantes dans le sous-dossier
    if images_batch:
        process_batch(images_batch, paths_batch, model, device_type)

if __name__ == '__main__':
    # Ajout de l'argument parser pour récupérer le répertoire passé en paramètre
    parser = argparse.ArgumentParser(description="Détection d'anime face et suppression des images vides")
    parser.add_argument("--directory", type=str, required=True, help="Répertoire à traiter")
    args = parser.parse_args()

    # On récupère le répertoire depuis l'argument --directory
    base_folder = args.directory

    # Parcourir uniquement les sous-dossiers de `base_folder` dans un ordre croissant
    for subdir in sorted(os.listdir(base_folder)):
        subfolder_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subfolder_path):
            print(f"Chargement et traitement du sous-dossier : {subfolder_path}")
            process_subfolder(
                subfolder_path,
                batch_size,
                model,
                device_type=device_type,
                num_processes=num_processes,
                pool_size=pool_size
            )
            print(f"Traitement du sous-dossier {subfolder_path} terminé.")
            print(f"Terminé le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
import os
import cv2
import torch
import psutil  # Importer psutil pour l'affinité CPU sur Windows
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configuration centrale des paramètres
device_type = 'cpu'  # Définir 'cpu' ou 'gpu' selon vos besoins
batch_size_gpu = 8  # Taille de lot pour le GPU
num_processes = 8  # Nombre de cœurs CPU pour le chargement des images
batch_size_cpu = num_processes  # Taille de lot pour le CPU

# Répertoire de base contenant les sous-dossiers
base_folder = r"T:\_SELECT\-HOKKAIDO GALS"

# Déterminer le périphérique (GPU ou CPU)
device = 'cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu'
print(f"Utilisation du périphérique : {device}")

# Configuration de l'affinité des cœurs CPU pour le script GPU
if device == 'cuda':
    p = psutil.Process()  # Obtenir le processus actuel
    p.cpu_affinity([0, 1, 2, 3, 16, 17, 18, 19])  # Utiliser les cœurs physiques 0-3 et leurs HT 16-19

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

# Fonction pour charger une image et redimensionner (CPU)
def load_and_resize_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.resize(img, target_size), image_path
    return None, image_path

# Fonction pour traiter un lot d'images (GPU ou CPU)
def process_batch(images_batch, paths_batch, model, device_type='gpu'):
    try:
        device = 'cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu'
        model.to(device)  # Envoyer le modèle sur le périphérique sélectionné
        
        if device == 'cuda':
            autocast_context = torch.amp.autocast('cuda')
        else:
            autocast_context = torch.no_grad()  # Sur CPU, utiliser torch.no_grad()

        with autocast_context:
            # Désactiver les logs de prédiction en ajoutant verbose=False
            results = model(images_batch, verbose=False)  # Appeler directement l'inférence sur les images (numpy.ndarray)

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
    
    except torch.cuda.OutOfMemoryError:
        print("Mémoire GPU insuffisante, basculement vers le CPU.")
        process_batch(images_batch, paths_batch, model, 'cpu')  # Retenter le traitement sur le CPU

# Fonction pour traiter un sous-dossier complet en parallèle (redimensionnement sur CPU)
def process_subfolder(subfolder, batch_size, model, device_type='gpu', num_processes=12):
    images_in_memory = []
    paths_in_memory = []

    # Charger toutes les images du sous-dossier en RAM en parallèle
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
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

    # Traiter les images par lots (inférence sur GPU ou CPU)
    images_batch = []
    paths_batch = []

    for img, image_path in zip(images_in_memory, paths_in_memory):
        images_batch.append(img)
        paths_batch.append(image_path)

        if len(images_batch) == batch_size:
            process_batch(images_batch, paths_batch, model, device_type)
            images_batch = []  # Réinitialiser le lot après traitement
            paths_batch = []

    # Traiter les images restantes dans le sous-dossier
    if images_batch:
        process_batch(images_batch, paths_batch, model, device_type)

# Utiliser `if __name__ == '__main__':` pour éviter les problèmes avec multiprocessing sur Windows
if __name__ == '__main__':
    # Parcourir uniquement les sous-dossiers de `base_folder` et traiter chaque sous-dossier séparément
    for subdir in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subfolder_path):
            print(f"Chargement et traitement du sous-dossier : {subfolder_path}")
            process_subfolder(subfolder_path, batch_size, model, device_type=device_type, num_processes=num_processes)
            print(f"Traitement du sous-dossier {subfolder_path} terminé.")
            print(f"Terminé le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

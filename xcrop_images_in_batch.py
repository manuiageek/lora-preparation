import os
import shutil
import cv2
import torch
import psutil
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------------------------------------
# Configuration générale
# ------------------------------------------------------------------------------------------
device_type = 'gpu'  # 'gpu' ou 'cpu'
num_processes = 16
batch_size_gpu = 16
batch_size_cpu = num_processes

# Pourcentage de marge autour de la bounding box
margin_ratio = 0.3  # 0.3 = 30%

# Déterminer le périphérique (GPU ou CPU)
device = 'cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu'
print(f"Utilisation du périphérique : {device}")

# Configuration de l'affinité des cœurs CPU (facultatif)
if device == 'cuda':
    p = psutil.Process()
    if num_processes == 8:
        p.cpu_affinity([0, 1, 2, 3, 17, 18, 19, 20])
    elif num_processes == 16:
        p.cpu_affinity([i for i in range(8)] + [i + 16 for i in range(8)])
    elif num_processes == 24:
        p.cpu_affinity([i for i in range(12)] + [i + 16 for i in range(12)])
    else:
        available_cores = list(range(psutil.cpu_count()))
        p.cpu_affinity(available_cores)

if device == 'cpu':
    torch.set_num_threads(num_processes)
    batch_size = batch_size_cpu
else:
    batch_size = batch_size_gpu

# ------------------------------------------------------------------------------------------
# Charger le modèle YOLOv8 animeface pré-entraîné
# ------------------------------------------------------------------------------------------
model_path = Path('models') / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))

# Taille de l'image pour réduire l'utilisation de la VRAM
target_size = (640, 640)

# ------------------------------------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------------------------------------
def load_images(image_path):
    """
    Charge l'image originale et sa version redimensionnée (640x640).
    Retourne (img_original, img_resized, image_path) ou (None, None, image_path) si échec.
    """
    img_original = cv2.imread(image_path)
    if img_original is None:
        return None, None, image_path

    img_resized = cv2.resize(img_original, target_size)
    return img_original, img_resized, image_path

def save_cropped_image(img_original, bbox, save_path):
    """
    Recadre l'image originale selon la bounding box (x1, y1, x2, y2) + marge, et sauvegarde.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img_original.shape[:2]

    # Sécuriser les bornes
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped_img = img_original[y1:y2, x1:x2]
    cv2.imwrite(save_path, cropped_img)
    print(f"Image recadrée sauvegardée : {save_path}")

# ------------------------------------------------------------------------------------------
# Traitement par batch
# ------------------------------------------------------------------------------------------
def process_batch(originals_batch, resized_batch, paths_batch, model, cropped_dir, device_type='gpu'):
    """
    Gère un lot d'images.
     - Pour chaque image, on détecte tous les visages anime.
     - S'il n'y en a aucun, on copie l'image originale.
     - S'il y en a plusieurs, on crée autant de fichiers recadrés que de visages.
    """
    try:
        device_local = 'cuda' if (device_type == 'gpu' and torch.cuda.is_available()) else 'cpu'
        model.to(device_local)

        if device_local == 'cuda':
            autocast_context = torch.amp.autocast('cuda')
        else:
            autocast_context = torch.no_grad()

        with autocast_context:
            results = model(resized_batch, verbose=False)

        for i, result in enumerate(results):
            # Récupération de TOUTES les bounding boxes
            faces = []
            for detection in result.boxes.data:
                # YOLO: (x1, y1, x2, y2, score, class)
                x1, y1, x2, y2, score, class_id = detection.tolist()
                if int(class_id) == 0:  # classe "visage anime"
                    faces.append((x1, y1, x2, y2, score))

            if len(faces) == 0:
                # Aucun visage => copie
                basename = os.path.basename(paths_batch[i])
                save_path = os.path.join(cropped_dir, basename)
                shutil.copy2(paths_batch[i], save_path)
                print(f"Aucun visage => copie : {save_path}")
            else:
                # Au moins un visage => on recadre pour chacun
                # On récupère l'image originale pour faire un recadrage propre
                img_original = originals_batch[i]
                h_orig, w_orig = img_original.shape[:2]

                # Échelles pour repasser du (640,640) à la taille d'origine
                scale_x = w_orig / float(target_size[0])
                scale_y = h_orig / float(target_size[1])

                # Nom de base du fichier
                basename = os.path.basename(paths_batch[i])
                name_no_ext, ext = os.path.splitext(basename)

                face_count = 1
                for (x1, y1, x2, y2, score) in faces:
                    x1_orig = x1 * scale_x
                    y1_orig = y1 * scale_y
                    x2_orig = x2 * scale_x
                    y2_orig = y2 * scale_y

                    # Calcul de la marge
                    bbox_width = x2_orig - x1_orig
                    bbox_height = y2_orig - y1_orig
                    marge_x = bbox_width * margin_ratio
                    marge_y = bbox_height * margin_ratio

                    x1_final = x1_orig - marge_x
                    y1_final = y1_orig - marge_y
                    x2_final = x2_orig + marge_x
                    y2_final = y2_orig + marge_y

                    # Construire un nom de fichier du type "nom_face1.jpg"
                    cropped_filename = f"{name_no_ext}_face{face_count}{ext}"
                    save_path = os.path.join(cropped_dir, cropped_filename)

                    # Sauvegarder le visage recadré
                    save_cropped_image(
                        img_original,
                        (x1_final, y1_final, x2_final, y2_final),
                        save_path
                    )
                    face_count += 1

    except torch.cuda.OutOfMemoryError:
        print("Mémoire GPU insuffisante, on bascule sur le CPU.")
        process_batch(originals_batch, resized_batch, paths_batch, model, cropped_dir, 'cpu')


def process_folder(folder, batch_size, model, cropped_dir, device_type='gpu', num_processes=12):
    """
    Récupère *toutes* les images (jpg, png, bmp...) directement dans 'folder'
    (pas dans les sous-dossiers).
    Les charge, puis les traite par lots, avec recadrage multiple si plusieurs visages.
    """
    originals_in_memory = []
    resized_in_memory = []
    paths_in_memory = []

    # Charger toutes les images en parallèle (CPU)
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(folder, filename)
                futures.append(executor.submit(load_images, image_path))

        # Récupérer les résultats
        for future in futures:
            img_orig, img_resized, image_path = future.result()
            if img_orig is not None and img_resized is not None:
                originals_in_memory.append(img_orig)
                resized_in_memory.append(img_resized)
                paths_in_memory.append(image_path)

    # Traiter par lots (GPU ou CPU)
    originals_batch = []
    resized_batch = []
    paths_batch = []

    for orig, resized, path_ in zip(originals_in_memory, resized_in_memory, paths_in_memory):
        originals_batch.append(orig)
        resized_batch.append(resized)
        paths_batch.append(path_)

        if len(originals_batch) == batch_size:
            process_batch(originals_batch, resized_batch, paths_batch, model, cropped_dir, device_type)
            originals_batch = []
            resized_batch = []
            paths_batch = []

    # Traiter les images restantes
    if originals_batch:
        process_batch(originals_batch, resized_batch, paths_batch, model, cropped_dir, device_type)

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    base_folder = input("Veuillez entrer le chemin du répertoire à traiter (images directement dedans) : ").strip()
    print(f"Traitement du dossier : {base_folder}")

    if not os.path.isdir(base_folder):
        print("Le chemin indiqué n'est pas un dossier valide.")
    else:
        # Supprimer le dossier "cropped" s'il existe déjà, puis le recréer
        cropped_dir = os.path.join(base_folder, "cropped")
        if os.path.exists(cropped_dir):
            print("Suppression du dossier 'cropped' existant...")
            shutil.rmtree(cropped_dir)
        os.makedirs(cropped_dir, exist_ok=True)

        # Lancer le traitement
        process_folder(
            base_folder,
            batch_size,
            model,
            cropped_dir,
            device_type=device_type,
            num_processes=num_processes
        )
        print(f"Traitement de {base_folder} terminé.")
        print(f"Terminé le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

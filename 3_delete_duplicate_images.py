import os
import psutil
from PIL import Image
import imagehash
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import argparse
import pickle

# Constantes globales pour le contrôle des valeurs
HASH_SIZE = 16  # Contrôle la précision du hachage perceptuel
THRESHOLD = 1   # Contrôle la tolérance pour considérer deux images comme doublons
SIZE_THRESHOLD = 1024 * 50  # Taille en octets pour la fenêtre glissante (50 Ko)
CACHE_FILE = r'image_hashes_cache.pkl'  # Fichier pour stocker le cache des hachages

def set_cpu_affinity():
    p = psutil.Process()  # Obtenir le processus actuel

    # Définissez le nombre de processus ici
    num_processes = 32  # Modifiez cette valeur selon vos besoins

    # Définir l'affinité des cœurs CPU en fonction de num_processes
    if num_processes == 8:
        # Utilisation des cœurs physiques 0 à 3, et SMT 17 à 20 (en évitant 16)
        cores = [0, 1, 2, 3, 17, 18, 19, 20]
    elif num_processes == 16:
        # Utiliser les 8 cœurs physiques (0 à 7) et leurs SMT (16 à 23)
        cores = [i for i in range(8)] + [i + 16 for i in range(8)]
    elif num_processes == 24:
        # Utiliser les 12 cœurs physiques (0 à 11) et leurs SMT (16 à 27)
        cores = [i for i in range(12)] + [i + 16 for i in range(12)]
    else:
        # Si aucun des cas ne correspond, utiliser tous les cœurs disponibles
        cores = list(range(psutil.cpu_count()))
    
    p.cpu_affinity(cores)
    print(f"Affinité des cœurs CPU définie sur : {cores}")

    return num_processes

def compute_image_hash(image_path):
    """Calcule le hachage perceptuel d'une image."""
    try:
        img_hash = imagehash.average_hash(Image.open(image_path), hash_size=HASH_SIZE)
        return (image_path, img_hash)
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return None

def load_hash_cache():
    """Charge le cache des hachages depuis le fichier."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_hash_cache(cache):
    """Sauvegarde le cache des hachages dans le fichier."""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def find_and_remove_duplicates(directory, hash_cache, num_processes):
    # Stocker les hachages des images
    hashes = {}
    duplicates = []

    # Récupérer tous les fichiers image du répertoire et trier par taille
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    image_files.sort(key=lambda x: os.path.getsize(x))

    # Obtenir les tailles des fichiers
    image_sizes = [os.path.getsize(f) for f in image_files]

    # Calculer les hachages des images en parallèle
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(compute_image_hash_with_cache, image_files, [hash_cache]*len(image_files))

    # Traiter les résultats avec une fenêtre glissante
    for idx, result in enumerate(results):
        if result:
            image_path, img_hash = result
            image_size = image_sizes[idx]

            # Fenêtre glissante : comparer avec les images de taille similaire
            window_start = idx
            while window_start > 0 and abs(image_sizes[window_start] - image_size) <= SIZE_THRESHOLD:
                window_start -= 1
            window_end = idx
            while window_end < len(image_sizes) - 1 and abs(image_sizes[window_end] - image_size) <= SIZE_THRESHOLD:
                window_end += 1

            # Comparer avec les hachages dans la fenêtre
            duplicate_found = False
            for other_idx in range(window_start, window_end + 1):
                if other_idx == idx:
                    continue
                other_image = image_files[other_idx]
                other_hash = hashes.get(other_image)
                if other_hash:
                    # Calculer la différence de hachage
                    if img_hash - other_hash < THRESHOLD:  # Tolère les petites différences
                        duplicates.append((image_path, other_image))

                        # Supprimer le doublon
                        os.remove(image_path)
                        print(f"Image en doublon supprimée : {image_path}")

                        duplicate_found = True
                        break

            # Ajouter l'image et son hachage si ce n'est pas un doublon
            if not duplicate_found:
                hashes[image_path] = img_hash

    return duplicates

def compute_image_hash_with_cache(image_path, hash_cache):
    """Calcule le hachage perceptuel d'une image en utilisant le cache."""
    if image_path in hash_cache:
        return (image_path, hash_cache[image_path])
    else:
        result = compute_image_hash(image_path)
        if result:
            _, img_hash = result
            hash_cache[image_path] = img_hash
        return result

def process_all_subdirectories(root_directory, num_processes):
    # Charger le cache des hachages
    hash_cache = load_hash_cache()

    # Récupérer tous les sous-répertoires dans l'ordre croissant
    for dirpath, dirnames, _ in os.walk(root_directory):
        # Trier les sous-répertoires dans l'ordre croissant
        dirnames.sort()

        # Traiter chaque répertoire
        print(f"Traitement du répertoire : {dirpath}")
        duplicates = find_and_remove_duplicates(dirpath, hash_cache, num_processes)

        if duplicates:
            print("\nRésumé des doublons supprimés :")
            for dup in duplicates:
                print(f"Image doublon supprimée : {dup[0]}")
        else:
            print("Aucun doublon trouvé dans ce répertoire.")

    # Sauvegarder le cache des hachages
    save_hash_cache(hash_cache)

    # Afficher l'heure de fin
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Traitement terminé le : {end_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supprimer les images en doublon dans un répertoire donné.")
    parser.add_argument(
        "--directory",
        type=str,
        default=r"T:\_SELECT\R_-TENCHI MUYO",
        help="Le chemin du répertoire contenant les fichiers à traiter."
    )

    args = parser.parse_args()

    # Définir l'affinité des cœurs CPU et obtenir le nombre de processus
    num_processes = set_cpu_affinity()

    # Appeler la fonction principale
    process_all_subdirectories(args.directory, num_processes)

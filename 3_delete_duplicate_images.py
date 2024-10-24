import os
from PIL import Image
import imagehash
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Spécifier le répertoire de base
directory = r"F:\1_TO_EXTRACT_1-2-3\TENCHI MUYO SPECIAL"

# Constantes globales pour le contrôle des valeurs
HASH_SIZE = 16  # Contrôle la précision du hachage perceptuel
THRESHOLD = 1   # Contrôle la tolérance pour considérer deux images comme doublons

def compute_image_hash(image_path):
    """Calcule le hachage perceptuel d'une image."""
    try:
        img_hash = imagehash.average_hash(Image.open(image_path), hash_size=HASH_SIZE)
        return (image_path, img_hash)
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return None

def find_and_remove_duplicates(directory):
    # Stocker les hachages des images
    hashes = {}
    duplicates = []

    # Récupérer tous les fichiers image du répertoire
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]

    # Calculer les hachages des images en parallèle
    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_image_hash, image_files)

    # Traiter les résultats
    for result in results:
        if result:
            image_path, img_hash = result
            
            # Vérifier si un hachage similaire existe déjà
            duplicate_found = False
            for existing_hash, existing_image in hashes.items():
                # Calculer la différence de hachage
                if img_hash - existing_hash < THRESHOLD:  # Tolère les petites différences
                    duplicates.append((image_path, existing_image))
                    
                    # Supprimer le doublon
                    os.remove(image_path)
                    print(f"Image en doublon supprimée : {image_path}")
                    
                    duplicate_found = True
                    break
            
            # Ajouter l'image et son hachage si ce n'est pas un doublon
            if not duplicate_found:
                hashes[img_hash] = image_path
    
    return duplicates

def process_all_subdirectories(root_directory):
    # Récupérer tous les sous-répertoires dans l'ordre croissant
    for dirpath, dirnames, _ in os.walk(root_directory):
        # Trier les sous-répertoires dans l'ordre croissant
        dirnames.sort()

        # Traiter chaque répertoire
        print(f"Traitement du répertoire : {dirpath}")
        duplicates = find_and_remove_duplicates(dirpath)

        if duplicates:
            print("\nRésumé des doublons supprimés :")
            for dup in duplicates:
                print(f"Image doublon supprimée : {dup[0]}")
        else:
            print("Aucun doublon trouvé dans ce répertoire.")

# Usage
process_all_subdirectories(directory)
print(f"Terminé le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

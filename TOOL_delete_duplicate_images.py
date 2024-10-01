import os
from PIL import Image
import imagehash

# Constantes globales pour le contrôle des valeurs
HASH_SIZE = 16  # Contrôle la précision du hachage perceptuel
THRESHOLD = 6   # Contrôle la tolérance pour considérer deux images comme doublons

def find_and_remove_duplicates(directory):
    # Stocker les hachages des images
    hashes = {}
    duplicates = []

    # Parcourir toutes les images dans le répertoire
    for image_name in os.listdir(directory):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(directory, image_name)
            # Calculer le hachage perceptuel de l'image
            img_hash = imagehash.average_hash(Image.open(image_path), hash_size=HASH_SIZE)
            
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
output_directory = r"/home/heimana/hleet/yt-dlp"
process_all_subdirectories(output_directory)

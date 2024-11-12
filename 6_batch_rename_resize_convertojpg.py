import os
from PIL import Image
import shutil

# Appliquer le redimensionnement si nécessaire
def resize_image(image, max_size=1024):
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        print(f"Redimensionnement de l'image à {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.LANCZOS), True
    else:
        return image, False

# Chemin du dossier contenant les images à traiter
folder_path = r"E:\AI_WORK\TRAINED_LORA\SPAWN\jade_sp\dataset"

# Vérifier si le dossier existe
if not os.path.isdir(folder_path):
    print(f"Le dossier {folder_path} n'existe pas.")
else:
    # Créer un répertoire "img_reworked" pour stocker les images
    output_folder = os.path.join(folder_path, 'img_reworked')
    resized_folder = os.path.join(output_folder, 'img_resized')

    # Supprimer le dossier "img_reworked" et son contenu s'il existe déjà
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"Dossier {output_folder} supprimé.")

    # Recréer les dossiers "img_reworked" et "img_resized"
    os.makedirs(resized_folder)
    print(f"Création du répertoire {output_folder} pour stocker les images modifiées.")
    print(f"Création du sous-répertoire {resized_folder} pour stocker les images redimensionnées.")

    # Lister tous les fichiers du dossier
    files = os.listdir(folder_path)

    # Créer un compteur qui commence à 1 pour nommer les fichiers
    counter = 1

    for file in files:
        file_path = os.path.join(folder_path, file)

        # Vérifier si c'est un fichier (et non un sous-dossier)
        if os.path.isfile(file_path):
            try:
                # Nouveau nom pour le fichier dans "img_reworked"
                new_filename = f"{str(counter).zfill(3)}.jpg"
                new_filepath = os.path.join(output_folder, new_filename)

                # Ouvrir l'image avec PIL
                with Image.open(file_path) as img:
                    # Appliquer le redimensionnement si l'image est trop grande
                    resized_img, was_resized = resize_image(img)

                    if was_resized:
                        # Sauvegarder l'image redimensionnée dans "img_resized"
                        resized_filepath = os.path.join(resized_folder, new_filename)
                        resized_img = resized_img.convert('RGB')  # Convertir en RGB pour éviter les erreurs lors de la sauvegarde en JPG
                        resized_img.save(resized_filepath, 'JPEG')
                        print(f"Image {file} redimensionnée et sauvegardée dans {resized_folder}.")
                    else:
                        # Sauvegarder une copie de l'image non redimensionnée dans "img_reworked"
                        img.convert('RGB').save(new_filepath, 'JPEG')
                        print(f"Fichier {file} copié dans {output_folder} sans redimensionnement.")

                # Incrémenter le compteur après chaque fichier converti
                counter += 1

            except Exception as e:
                print(f"Impossible de traiter le fichier {file}: {e}")
        else:
            print(f"'{file}' est un dossier, passage au fichier suivant.")

    print(f"Traitement terminé. {counter - 1} fichiers convertis.")

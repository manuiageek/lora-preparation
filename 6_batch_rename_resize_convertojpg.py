import os
import sys
from PIL import Image
import shutil

# Appliquer le redimensionnement si nécessaire (vers le bas et vers le haut)
def resize_image(image, max_size=1024):
    width, height = image.size
    # Calculer le ratio pour redimensionner l'image tout en maintenant le rapport d'aspect
    ratio = min(max_size / width, max_size / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Vérifier si le redimensionnement est nécessaire
    if new_width != width or new_height != height:
        print(f"Redimensionnement de l'image à {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.LANCZOS), True
    else:
        return image, False

# Vérifier les arguments passés au script
keep_names = False
if len(sys.argv) > 1 and sys.argv[1].lower() == "keep":
    keep_names = True
    print("Mode 'keep' activé : les noms de fichiers d'origine seront conservés.")
else:
    print("Mode standard : les fichiers seront renommés en 001, 002, etc.")

# Chemins fixes pour le dossier des images à traiter et le dossier de sortie
input_folder = r"G:\GIGAPIXELS\UPSCALED"
output_folder = r"G:\GIGAPIXELS\img"

# Supprimer le dossier de sortie et son contenu s'il existe déjà
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    print(f"Dossier {output_folder} supprimé.")

# Recréer le dossier de sortie
os.makedirs(output_folder)
print(f"Création du répertoire {output_folder} pour stocker les images modifiées.")

# Lister tous les fichiers du dossier d'entrée
files = os.listdir(input_folder)

# Créer un compteur qui commence à 1 pour nommer les fichiers
counter = 1

for file in files:
    file_path = os.path.join(input_folder, file)

    # Vérifier si c'est un fichier (et non un sous-dossier)
    if os.path.isfile(file_path):
        try:
            if keep_names:
                # Conserver le nom de fichier original
                new_filename = os.path.splitext(file)[0] + ".jpg"
            else:
                # Nouveau nom pour le fichier dans le dossier de sortie
                new_filename = f"{str(counter).zfill(3)}.jpg"
            
            new_filepath = os.path.join(output_folder, new_filename)

            # Ouvrir l'image avec PIL
            with Image.open(file_path) as img:
                # Appliquer le redimensionnement si nécessaire (vers le bas ou le haut)
                resized_img, was_resized = resize_image(img)

                # Sauvegarder l'image redimensionnée ou non dans le dossier de sortie
                resized_img = resized_img.convert('RGB')  # Convertir en RGB pour éviter les erreurs lors de la sauvegarde en JPG
                resized_img.save(new_filepath, 'JPEG')

            if not keep_names:
                # Incrémenter le compteur après chaque fichier converti
                counter += 1

        except Exception as e:
            print(f"Impossible de traiter le fichier {file}: {e}")
    else:
        print(f"'{file}' est un dossier, passage au fichier suivant.")

message = "Tous les fichiers ont été conservés avec leurs noms d'origine." if keep_names else f"{counter - 1} fichiers convertis."
print(f"Traitement terminé. {message}")
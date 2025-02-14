import os

def main():
    # Demander le chemin du dossier principal
    base_folder = input("Entrez le chemin du dossier principal : ").strip()

    # Construire le chemin du dossier img
    img_folder = os.path.join(base_folder, 'img')

    # Vérifier si le dossier img existe
    if not os.path.exists(img_folder) or not os.path.isdir(img_folder):
        print(f"Le dossier 'img' n'existe pas à l'emplacement attendu : {img_folder}")
        return

    # Vérifier si des fichiers .txt existent
    txt_files = [f for f in os.listdir(img_folder) if f.endswith('.txt')]

    if not txt_files:
        print(f"Aucun fichier .txt trouvé dans {img_folder}")
        return

    output_file = os.path.join(os.path.dirname(img_folder), 'caption.txt')

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for index, file_name in enumerate(txt_files):
                file_path = os.path.join(img_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    if index < len(txt_files) - 1:
                        outfile.write("\n\n")

        print(f"Traitement terminé. Le fichier caption.txt a été créé avec succès à l'emplacement {output_file}")

    except Exception as e:
        print(f"Erreur : Impossible d'écrire dans {output_file}. Détails : {e}")


if __name__ == "__main__":
    main()

import os
import argparse

def main(base_folder):
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

    # Lire et concaténer le contenu des fichiers .txt
    all_words = []
    first_word = None

    try:
        for index, file_name in enumerate(txt_files):
            file_path = os.path.join(img_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().split()
                if content:
                    if first_word is None:
                        first_word = content[0]
                    all_words.extend(content)

        # Supprimer les doublons tout en gardant le premier mot en premier
        unique_words = list(dict.fromkeys(all_words))
        if first_word in unique_words:
            unique_words.remove(first_word)
        unique_words.insert(0, first_word)

        # Écrire le contenu unique dans caption.txt
        output_file = os.path.join(os.path.dirname(img_folder), 'caption.txt')
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(' '.join(unique_words))

        print(f"Le fichier {output_file} a été créé avec succès, sans doublons.")

    except Exception as e:
        print(f"Erreur : Impossible de traiter les fichiers. Détails : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concaténer le contenu des fichiers .txt dans un répertoire donné.")
    parser.add_argument(
        "--directory", 
        type=str, 
        required=True, 
        help="Le chemin du répertoire principal contenant le dossier 'img'."
    )
    
    args = parser.parse_args()
    main(args.directory)
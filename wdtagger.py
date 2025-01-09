import onnxruntime
import torch  # Juste pour réutiliser transforms (sinon tu peux t'en passer)
from torchvision import transforms
from PIL import Image
import numpy as np
import csv

so = onnxruntime.SessionOptions()
so.log_severity_level = 3  # 3 = ERROR seulement, 2 = WARNING, 1 = INFO

def load_image_for_wd14(image_path):
    tfm = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    x_torch = tfm(img)                 # Tensor shape: (3, 448, 448)
    x_torch = x_torch.unsqueeze(0)     # => (1, 3, 448, 448)
    
    x_np = x_torch.numpy()             # Conversion en numpy
    # On permute pour passer de (1, C, H, W) -> (1, H, W, C)
    x_np = np.transpose(x_np, (0, 2, 3, 1))  # => (1, 448, 448, 3)

    return x_np


def load_tags_csv(csv_path):
    """
    Charge la liste des tags depuis wd-swinv2-tagger-v3.csv
    Format: "0, tag_name"
    On renvoie un simple tableau de tags dans l'ordre.
    """
    tags = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # row[0] = index, row[1] = tag_name
            if len(row) >= 2:
                tags.append(row[1])
    return tags

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_tags_onnx(
    model_onnx_path,
    image_path,
    csv_tags_path,
    threshold=0.35,
    character_threshold=0.85
):
    # 1) Charger la session ONNX
    session = onnxruntime.InferenceSession(model_onnx_path, providers=['CUDAExecutionProvider'],sess_options=so)
    # si tu as la version GPU, tu peux tenter : providers=['CUDAExecutionProvider']

    # 2) Charger l'image et prétraiter
    input_data = load_image_for_wd14(image_path)  # shape: (1,3,448,448)

    # 3) Récupérer le nom de l'entrée (souvent "input" ou "x")
    input_name = session.get_inputs()[0].name
    # Récupérer le nom de la sortie (souvent "output" ou "logits")
    output_name = session.get_outputs()[0].name

    # 4) Lancer l’inférence
    #    onnxruntime attend un dico {nom_de_l_entree: matrice_numpy}
    outputs = session.run([output_name], {input_name: input_data})
    # outputs est une liste ; on récupère le premier (logits)
    logits = outputs[0]  # shape: (1, 10861) en théorie

    # 5) Sigmoïde pour transformer en probas
    probs = sigmoid(logits)[0]  # on enlève la dimension batch => (10861,)

    # 6) Charger la liste des tags depuis le CSV
    all_tags = load_tags_csv(csv_tags_path)  # liste de ~10861 tags

    # 7) Associer probas <-> tags + filtrage
    results = []
    for tag_name, p in zip(all_tags, probs):
        # Filtrage double threshold
        if "character" in tag_name.lower():
            if p >= character_threshold:
                results.append((tag_name, p))
        else:
            if p >= threshold:
                results.append((tag_name, p))

    # 8) Tri par probabilité décroissante
    results.sort(key=lambda x: x[1], reverse=True)

    # Si tu veux tout bonnement n’afficher que les tags > 0.5
    filtered = [(t, prob) for (t, prob) in results if prob >= 0.55]

    return filtered

if __name__ == "__main__":
    # Chemins
    model_onnx_path = r"./models/wdtagger/wd-swinv2-tagger-v3.onnx"
    csv_tags_path   = r"./models/wdtagger/wd-swinv2-tagger-v3.csv"
    test_image      = r"./images/ppeach_001.jpg"

    # Lancer la prédiction
    preds = predict_tags_onnx(
        model_onnx_path,
        test_image,
        csv_tags_path,
        threshold=0.20,
        character_threshold=0.85
    )

    # Afficher
    for tag, p in preds:
        print(f"{tag}: {p:.3f}")

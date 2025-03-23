import json
import requests

# URL de l'API ComfyUI (vérifie que c'est le bon endpoint)
url = "http://127.0.0.1:8188/prompt"
headers = {"Content-Type": "application/json"}

# Charge le workflow depuis ton fichier JSON
with open("CAPTION_API.json", "r", encoding="utf-8") as f:
    workflow = json.load(f)

# MODIFICATION DES INFORMATIONS PARAMETRES #
############################################
## TAGS_EXCLUDED_PARAM
workflow["105"]["inputs"]["Text"] = "1girl,1boy,solo,"

## KEYWORD PARAM
workflow["67"]["inputs"]["Text"] = "lucy"

## PATH_IMG_PARAM
workflow["94"]["inputs"]["value"] = r"E:\AI_WORK\TRAINED_LORA\FAIRY TAIL 100 YEARS QUEST\_TEST\lucy_ft100yq\ref\lucy-heartfilia-1.jpg"

## PATH_PARAM
workflow["103"]["inputs"]["value"] = r"E:\AI_WORK\TRAINED_LORA\FAIRY TAIL 100 YEARS QUEST\_TEST\lucy_ft100yq\ref"

# SEND #
########
payload = {"prompt": workflow}
response = requests.post(url, json=payload, headers=headers)
print(response.json())

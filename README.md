Some python scripts (written with the help of chatgpt) that I use to prepare LoRA (Low Rank Adaptation) training for anime characters.
- 1_rename_mkv.py : it renames the titles of mkv episode to "01", "02" and so forth
- 2_ffmpeg_extr_jpg.py : extract frames from mkv with ffmpeg tool
- 3_countfiles_ssfolders.py : counting files from subfolders to check the export in jpg
- 4_detect_person_plus_delete.py : it loads yolov8x6_animeface model in order to check if AI can find a "face" inside loop of images in a folder. Deleting any image that doesn't contain "anime face" in the frame to save disk space.
- 5_character_categorisation.py : https://github.com/KichangKim/DeepDanbooru/releases
- 6_batch_rename_resize_convertojpg.py
- 7_Comfyui_PREPLORA.json : workflow for comfyui

GPU ACCELERATION : 
For gpu acceleration, use miniconda : https://docs.anaconda.com/miniconda/
PYTORCH CONDA GUIDE :
conda create -n ptorch_env python=3.9
conda activate ptorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy opencv-python ultralytics

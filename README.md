Some python scripts (written with the help of chatgpt) that I use to prepare LoRA (Low Rank Adaptation) training for anime characters.
- 1_rename_mkv.py : it renames the titles of mkv episode to "01", "02" and so forth
- 2_ffmpeg_extr_jpg.py : extract frames from mkv with ffmpeg tool
- 3_delete_duplicate_images.py : pip install imagehash pillow
- 4_detect_person_or_delete.py : it loads yolov8x6_animeface model in order to check if AI can find a "face" inside loop of images in a folder. Deleting any image that doesn't contain "anime face" in the frame to save disk space.
- 5_character_categorisation.py : https://github.com/KichangKim/DeepDanbooru/releases
- 6_batch_rename_resize_convertojpg.py
- 7_Comfyui_PREPLORA.json : workflow for comfyui

NOTE : the 5_character_categorisation.py can not detect background and is faster because of tensorflow. So you could skip the 4_detect_person_or_delete.py.

GPU ACCELERATION : 
For gpu acceleration, use miniconda : https://docs.anaconda.com/miniconda/
PYTORCH CONDA GUIDE :
conda create -n ptorch_env python=3.9
conda activate ptorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy opencv-python ultralytics 

TENSORFLOW CONDA GUIDE :
conda create -n tflow_env python=3.10
conda activate tflow_env
conda install cudatoolkit=11.8 cudnn=8.6 -c conda-forge
pip install tensorflow==2.10.0
pip install numpy pillow tensorflow-io deepdanbooru psutil

check TensorFlow GPU :
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

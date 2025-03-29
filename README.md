## REQUIREMENTS :

### GPU ACCELERATION :

For gpu acceleration, use miniconda : https://docs.anaconda.com/miniconda/
PYTORCH CONDA GUIDE :
conda create -n ptorch_env python=3.9
conda activate ptorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy opencv-python ultralytics

### TENSORFLOW CONDA GUIDE :

conda create -n tflow_env python=3.10
conda activate tflow_env
conda install cudatoolkit=11.8 cudnn=8.6 -c conda-forge
pip install tensorflow==2.10.0
pip install numpy pillow tensorflow-io deepdanbooru psutil

check TensorFlow GPU :
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### NODEJS :

npm init -y
npm install puppeteer
npm install node-fetch
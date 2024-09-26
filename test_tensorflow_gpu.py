import tensorflow as tf

print("Version de TensorFlow :", tf.__version__)

# Vérifier si TensorFlow est construit avec CUDA
if tf.test.is_built_with_cuda():
    print("TensorFlow est construit avec CUDA")
else:
    print("TensorFlow n'est pas construit avec CUDA")

# Lister les GPU disponibles
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs détectés :")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("Aucun GPU détecté")

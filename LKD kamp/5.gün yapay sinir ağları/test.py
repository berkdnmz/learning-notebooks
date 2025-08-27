import tensorflow as tf

print("TensorFlow versiyonu:", tf.__version__)
print("GPU var mı:", tf.config.list_physical_devices('GPU'))
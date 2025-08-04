# utils.py
from PIL import Image
import tensorflow as tf
import io

def preprocess_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image
# model_loader.py
import tensorflow as tf
from tensorflow.keras import layers as L
import efficientnet.tfkeras as efn

def load_model(weights_path: str):
    model = tf.keras.Sequential([
        efn.EfficientNetB2(
            input_shape=(256, 256, 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(1024, activation='relu'),
        L.Dropout(0.3),
        L.Dense(512, activation='relu'),
        L.Dropout(0.2),
        L.Dense(256, activation='relu'),
        L.Dropout(0.2),
        L.Dense(128, activation='relu'),
        L.Dropout(0.1),
        L.Dense(1, activation='sigmoid')
    ])
    model.load_weights(weights_path)
    return model
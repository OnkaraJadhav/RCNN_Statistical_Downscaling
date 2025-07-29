"""
model.py

Summary:
    Defines a U-Net architecture using TensorFlow and Keras layers.
    Supports input masking and skip connections for dense regression.

Inputs:
    - Input feature tensor of shape (H, W, C)
    - Binary mask of shape (H, W, 1) to ignore land points

Outputs:
    - A compiled U-Net model instance with masked output

Functions:
    - build_unet(input_shape): returns Keras model

Used In:
    - train.py, test_full_inference.py
"""


# model.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose,
    MaxPooling2D, Concatenate,
    Multiply, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model

def build_unet(input_shape):
    inputs  = Input(shape=input_shape,       name='main_input')
    mask_in = Input(shape=input_shape[:2]+(1,), name='mask_input')

    def conv_block(x, filters):
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x); x = tf.nn.relu(x)
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x); x = tf.nn.relu(x)
        return x

    # Encoder
    c1 = conv_block(inputs, 64);  p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, 128);     p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, 256);     p3 = MaxPooling2D()(c3)
    c4 = conv_block(p3, 512);     c4 = Dropout(0.2)(c4)

    # Decoder
    u3 = Conv2DTranspose(256, 2, strides=2, padding='same')(c4)
    u3 = Concatenate()([u3, c3])
    c5 = conv_block(u3, 256);     c5 = Dropout(0.1)(c5)

    u2 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u2 = Concatenate()([u2, c2])
    c6 = conv_block(u2, 128)

    u1 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u1 = Concatenate()([u1, c1])
    c7 = conv_block(u1, 64)

    outputs = Conv2D(1, 1, activation='linear')(c7)
    masked  = Multiply()([outputs, mask_in])

    return Model([inputs, mask_in], masked)

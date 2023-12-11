from typing import Union

import tensorflow as tf
import keras
import numpy as np


class CowModel(keras.Model):
    def __init__(
        self,
        output_dim: int,
    ) -> None:
        """
        `output_dim`: The number of classes (labels).
        """
        super().__init__()
        self.conv1d_128: keras.layers = keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="valid",
            dtype=tf.float32,
        )
        self.conv1d_64: keras.layers = keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="valid",
            dtype=tf.float32,
        )
        self.conv1d_32: keras.layers = keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="valid",
            dtype=tf.float32,
        )
        self.batch_norm: keras.layers = keras.layers.BatchNormalization()
        self.conv1d_transpose_16: keras.layers = tf.keras.layers.Conv1DTranspose(
            filters=16,
            kernel_size=3,
            padding="valid",
            dtype=tf.float32,
        )
        self.conv1d_transpose_8: keras.layers = tf.keras.layers.Conv1DTranspose(
            filters=8,
            kernel_size=3,
            padding="valid",
            dtype=tf.float32,
        )
        self.conv1d_transpose_4: keras.layers = tf.keras.layers.Conv1DTranspose(
            filters=output_dim,
            kernel_size=3,
            padding="valid",
            dtype=tf.float32,
        )

        # self.conv1 = keras.layers.Conv2D(32, 3, activation="relu")
        # self.flatten = keras.layers.Flatten()
        # self.d1 = keras.layers.Dense(128, activation="relu")
        # self.d2 = keras.layers.Dense(10)

    def call(
        self,
        x: Union[
            tf.data.Dataset,
            tf.Tensor,
            np.ndarray,
        ],
    ) -> tf.Tensor:
        x: tf.Tensor = self.conv1d_128(x)
        x = self.batch_norm(x)
        x = self.conv1d_64(x)
        # x = self.batch_norm(x)
        x = self.conv1d_32(x)
        # x = self.batch_norm(x)
        x = self.conv1d_transpose_16(x)
        x = self.conv1d_transpose_8(x)
        return self.conv1d_transpose_4(x)

        # x = self.conv1(x)
        # x = self.flatten(x)
        # x = self.d1(x)
        # return self.d2(x)

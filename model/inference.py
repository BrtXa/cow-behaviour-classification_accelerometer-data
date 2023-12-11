from typing import Union

import copy
import tensorflow as tf
import keras
import numpy as np


class Inference:
    def __init__(
        self,
        model: keras.Model,
        loss_function: keras.losses.Loss,
        loss_metric: keras.metrics.Metric,
        optimizer: keras.optimizers.Optimizer,
        accuracy: keras.metrics.Metric,
        batch_size: int,
    ) -> None:
        self.model: keras.Model = model
        self.loss_function: keras.losses.Loss = loss_function
        self.train_loss: keras.metrics.Metric = copy.deepcopy(loss_metric)
        self.test_loss: keras.metrics.Metric = copy.deepcopy(loss_metric)
        self.optimizer: keras.optimizers.Optimizer = optimizer
        self.train_accuracy: keras.metrics.Metric = copy.deepcopy(accuracy)
        self.test_accuracy: keras.metrics.Metric = copy.deepcopy(accuracy)

    def train_val(
        self,
        train_data: Union[
            tf.data.Dataset,
            tf.Tensor,
            np.ndarray,
        ],
        val_data: Union[
            tf.data.Dataset,
            tf.Tensor,
            np.ndarray,
        ],
        epoch: int = 6,
    ) -> dict[str, list]:
        metrics: dict[str, list] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for e in range(epoch):
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.test_loss.reset_state()
            self.test_accuracy.reset_state()

            for data, labels in train_data:
                self.train_step(
                    data=data,
                    labels=labels,
                )

            for data, labels in val_data:
                self.test_step(
                    data=data,
                    labels=labels,
                )

            # print(
            #     "epoch: {} - train loss: {} - train acc: {} - test loss: {} - test acc: {}".format(
            #         e,
            #         self.train_loss.result(),
            #         self.train_accuracy.result(),
            #         self.test_loss.result(),
            #         self.test_accuracy.result(),
            #     )
            # )
            metrics["train_loss"].append(self.train_loss.result())
            metrics["train_accuracy"].append(self.train_accuracy.result())
            metrics["val_loss"].append(self.test_loss.result())
            metrics["val_accuracy"].append(self.test_accuracy.result())

        return metrics

    def test(
        self,
        test_data: Union[
            tf.data.Dataset,
            tf.Tensor,
            np.ndarray,
        ],
    ):
        metrics: dict[str, list] = {
            "test_loss": [],
            "test_accuracy": [],
        }

        self.test_loss.reset_state()
        self.test_accuracy.reset_state()
        for data, labels in test_data:
            self.test_step(
                data=data,
                labels=labels,
            )
        # print(
        #     "test loss: {} - test acc: {}".format(
        #         self.test_loss.result(),
        #         self.test_accuracy.result(),
        #     )
        # )
        metrics["test_loss"].append(self.test_loss.result())
        metrics["test_accuracy"].append(self.test_accuracy.result())

        return metrics

    @tf.function
    def train_step(
        self,
        data: tf.Tensor,
        labels: tf.Tensor,
    ) -> None:
        with tf.GradientTape() as tape:
            preds: tf.Tensor = self.model(
                data,
                training=True,
            )
            loss: tf.Tensor = self.loss_function(labels, preds)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, preds)

    @tf.function
    def test_step(
        self,
        data: tf.Tensor,
        labels: tf.Tensor,
    ) -> None:
        preds: tf.Tensor = self.model(
            data,
            training=False,
        )
        t_loss: tf.Tensor = self.loss_function(labels, preds)

        self.test_loss(t_loss)
        self.test_accuracy(labels, preds)

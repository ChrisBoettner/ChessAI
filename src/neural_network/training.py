#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:09:21 2023

@author: chris
"""
from dataclasses import dataclass
from typing import Any, Optional

import tensorflow as tf

from neural_network.architecture import AlphaZeroNetwork
from neural_network.loss import AlphaZeroLoss
from neural_network.metrics import AlphaZeroLossMetric
from neural_network.preprocessing import prepare_dataset
from utilities import DataHandler, load_hyperparameter


class Trainer:
    """
    A class that initializes hyperparameters, loads data, prepares the dataset,
    creates, and trains the model.
    """

    def __init__(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss_function: Optional[tf.keras.losses.Loss] = None,
        model: Optional[tf.keras.Model] = None,
        train_dataset: Optional[tf.data.Dataset] = None,
        val_dataset: Optional[tf.data.Dataset] = None,
        filename: str = "game_data",
        num_datapoints: Optional[int] = None,
    ):
        """
        Initializes the Trainer.

        Parameters
        ----------
        optimizer : Optimizer, optional
            The optimizer to use for training the model.
            If None, tf.keras.optimizers.Adam will be used.
        loss_function : Loss function, optional
            The loss function for training the model.
            If None, AlphaZeroLoss will be used.
        model : tf.keras.Model, optional
            The model to train.
            If None, AlphaZeroNetwork will be created.
        train_dataset : Dataset, optional
            The dataset to train on.
            If None, the dataset will be loaded from a DataHandler.
        val_dataset : Dataset, optional
            The dataset used for validation.
            If None, the dataset will be loaded from a DataHandler.
        filename : str, optional
            Name of file data is laoded from. The default is "game_data".
        num_datapoints : int, optional
            Number of lines to read from the file. If None, read all lines.
            The default is None.
        """

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print("Setting Hyperparameter...")
        self.hyperparameter = NNHyperparameter()
        load_hyperparameter(self.hyperparameter, "NEURAL_NETWORK")

        if (self.train_dataset is None) or (self.val_dataset is None):
            print("Creating Dataset...")
            boards, policies, values = DataHandler().load(
                filename=filename, num_datapoints=num_datapoints
            )
            self.train_dataset, self.val_dataset = prepare_dataset(
                boards,
                policies,
                values,
                buffer_size=self.hyperparameter.BUFFER_SIZE,
                batch_size=self.hyperparameter.BATCH_SIZE,
            )

        if self.model is None:
            print("Creating Model...")
            self.model = AlphaZeroNetwork(
                self.hyperparameter.FILTERS, self.hyperparameter.RES_LAYERS
            )

        if self.optimizer is None:
            print("Creating Optimizer...")
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.hyperparameter.LEARNING_RATE
            )

        if self.loss_function is None:
            print("Creating Loss Function Instance...")
            self.loss_function = AlphaZeroLoss(self.model, self.hyperparameter.L2_REG)
        print("Done.\n")

    def train(self, **kwargs: Any) -> tf.keras.Model:
        """
        Trains the model.

        Returns
        -------
        tf.keras.Model
            The trained AlphaZeroNetwork model.

        """
        self.train_loop(
            epochs=self.hyperparameter.EPOCHS,
            patience=self.hyperparameter.PATIENCE,
            **kwargs,
        )
        return self.model

    def train_loop(
        self,
        epochs: int,
        patience: int,
        checkpoint_dir: str = "model_checkpoints/",
        update_freq: int = 50,
    ) -> None:
        """
        Training loop for the model with checkpoints and early stopping.

        Parameters
        ----------
        epochs : int
            Number of epochs to train for.
        patience: int, optional
            Number of epochs with no improvement after which training will be stopped.
            The default is 5.
        checkpoint_dir : str, optional
            Directory where to save checkpoints. The default is "model_checkpoints/".
        update_freq : int, optional
            The frequency (in terms of number of batches) to print progress updates.
            The default is 50.

        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        train_loss_metric = AlphaZeroLossMetric(name="train_loss")
        val_loss_metric = AlphaZeroLossMetric(name="train_loss")

        # Set up checkpoint objects
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        for epoch in range(epochs):
            print(f"Start of epoch {epoch}")

            # Training loop
            assert self.train_dataset is not None
            for batch_num, batch in enumerate(self.train_dataset):
                self.train_step(
                    batch["boards"],
                    batch["policies"],
                    batch["masks"],
                    batch["values"],
                )
                train_loss_metric.update_state(self.loss_function)

                # Print update every 'update_freq' batches
                if (batch_num + 1) % update_freq == 0:
                    (
                        total_loss_avg,
                        policy_loss_avg,
                        value_loss_avg,
                        l2_loss_avg,
                    ) = train_loss_metric.result()
                    print(
                        f"Epoch {epoch}, Batch {batch_num+1},"
                        f" Training Loss: {total_loss_avg:.2f} |"
                        f" Policy Loss: {policy_loss_avg:.2f},"
                        f" Value Loss: {value_loss_avg:.2f},"
                        f" L2 Loss: {l2_loss_avg:.2f}"
                    )

            # Evaluation loop
            assert self.val_dataset is not None
            for batch in self.val_dataset:
                self.evaluation_step(
                    batch["boards"],
                    batch["policies"],
                    batch["masks"],
                    batch["values"],
                )
                val_loss_metric.update_state(self.loss_function)

            # End of epoch
            train_loss = train_loss_metric.result()[0]
            val_loss = val_loss_metric.result()[0]
            print(
                f"Epoch {epoch} - train loss: {train_loss:.2f}, "
                f"val loss: {val_loss:.2f}"
            )

            # Check early stopping condition
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model so far
                manager.save()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping due to no improvement in validation loss.")
                    break

            # Reset metrics for the next epoch
            train_loss_metric.reset_states()
            val_loss_metric.reset_states()
        print("Maximum number of epochs reached.")
        # Load the best model at the end of training
        checkpoint.restore(manager.latest_checkpoint)

    def evaluation_step(
        self,
        boards: tf.Tensor,
        policies: tf.Tensor,
        masks: tf.Tensor,
        values: tf.Tensor,
    ) -> float:
        """
        Perform one evaluation step. (Forward pass and calculate loss.)

        Parameters
        ----------
        boards : tf.Tensor
            The board states, converted to tensors.
        policies : tf.Tensor
            The true policies tensors.
        masks : tf.Tensor
            The valid move masks.
        values : tf.Tensor
            The true board state values.

        Returns
        -------
        float
            Computed total AlphaZero loss with optional L2 regularization.

        """
        assert isinstance(self.loss_function, tf.keras.losses.Loss)
        assert isinstance(self.model, tf.keras.Model)
        policy_estimates, value_estimates = self.model(boards, masks)
        loss = self.loss_function(
            (policies, values), (policy_estimates, value_estimates)
        )
        return loss

    def train_step(
        self,
        boards: tf.Tensor,
        policies: tf.Tensor,
        masks: tf.Tensor,
        values: tf.Tensor,
    ) -> float:
        """
        Perform one training step. (Forward pass, calculate loss, then gradients,
        then apply gradients.)

        Parameters
        ----------
        boards : tf.Tensor
            The board states, converted to tensors.
        policies : tf.Tensor
            The true policies tensors.
        masks : tf.Tensor
            The valid move masks.
        values : tf.Tensor
            The true board state values.

        Returns
        -------
        float
            Computed total AlphaZero loss with optional L2 regularization.

        """
        assert isinstance(self.optimizer, tf.keras.optimizers.Optimizer)
        assert isinstance(self.model, tf.keras.Model)
        with tf.GradientTape() as tape:
            loss = self.evaluation_step(boards, policies, masks, values)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss


@dataclass
class NNHyperparameter:
    """
    Data class to store hyperparameter.

    """

    BUFFER_SIZE: int = 0
    BATCH_SIZE: int = 0
    LEARNING_RATE: float = 0
    L2_REG: float = 0
    EPOCHS: int = 0
    FILTERS: int = 0
    RES_LAYERS: int = 0
    PATIENCE: int = 0

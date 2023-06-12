#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:09:21 2023

@author: chris
"""
from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from neural_network.architecture import AlphaZeroNetwork
from neural_network.loss import AlphaZeroLoss
from neural_network.metrics import AlphaZeroLossComponents
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
        dataset: Optional[tf.data.Dataset] = None,
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
        dataset : Dataset, optional
            The dataset to train on.
            If None, the dataset will be loaded from a DataHandler.
        """

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model
        self.dataset = dataset

        print("Setting Hyperparameter...")
        self.hyperparameter = NNHyperparameter()
        load_hyperparameter(self.hyperparameter, "NEURAL_NETWORK")

        if self.dataset is None:
            print("Creating Dataset...")
            boards, policies, values = DataHandler().load()
            self.dataset, self.val_dataset = prepare_dataset(
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

    def train(self) -> tf.keras.Model:
        """
        Trains the model.

        Returns
        -------
        tf.keras.Model
            The trained AlphaZeroNetwork model.

        """
        train(
            model=self.model,
            epochs=self.hyperparameter.EPOCHS,
            train_dataset=self.dataset,
            val_dataset=self.val_dataset,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
        )
        return self.model


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


def evaluation_step(
    model: tf.keras.Model,
    loss_function: tf.keras.losses.Loss,
    boards: tf.Tensor,
    policies: tf.Tensor,
    masks: tf.Tensor,
    values: tf.Tensor,
) -> float:
    """
    Perform one evaluation step.

    Parameters
    ----------
    model : tf.keras.Model
        The neural network model.
    loss_function : tf.keras.losses.Loss
        The loss function.
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
    policy_estimates, value_estimates = model(boards, masks)
    loss = loss_function((policies, values), (policy_estimates, value_estimates))
    return loss


def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_function: tf.keras.losses.Loss,
    boards: tf.Tensor,
    policies: tf.Tensor,
    masks: tf.Tensor,
    values: tf.Tensor,
) -> float:
    """
    Perform one training step. (Calculate loss, then gradients, then apply gradients.)

    Parameters
    ----------
    model : tf.keras.Model
        The neural network model.
    loss_function : tf.keras.losses.Loss
        The loss function.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer.
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
    with tf.GradientTape() as tape:
        loss = evaluation_step(model, loss_function, boards, policies, masks, values)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def train(
    model: tf.keras.Model,
    epochs: int,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_function: tf.keras.losses.Loss,
    checkpoint_dir: str = "model_checkpoints/",
    patience: int = 5,
    update_freq: int = 10,
) -> None:
    """
    Training loop for the model with checkpoints and early stopping.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be trained.
    epochs : int
        Number of epochs to train for.
    train_dataset : tf.data.Dataset
        The training dataset.
    val_dataset : tf.data.Dataset
        The validation dataset.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use.
    loss_function : tf.keras.losses.Loss
        The loss function to use.
    checkpoint_dir : str, optional
        Directory where to save checkpoints. The default is "model_checkpoints/".
    patience: int, optional
        Number of epochs with no improvement after which training will be stopped.
        The default is 5.
    update_freq : int, optional
        The frequency (in terms of number of batches) to print progress updates.
        The default is 10.

    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_loss_metric = AlphaZeroLossComponents(name="train_loss")
    val_loss_metric = AlphaZeroLossComponents(name="train_loss")

    # Set up checkpoint objects
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")

        # Training loop
        for batch_num, batch in enumerate(train_dataset):
            boards, policies, masks, values = batch
            train_step(model, optimizer, loss_function, boards, policies, masks, values)
            train_loss_metric.update_state(loss_function)

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
                    f" Training Loss: {total_loss_avg:.2f},"
                    f" Policy Loss: {policy_loss_avg:.2f},"
                    f" Value Loss: {value_loss_avg:.2f},"
                    f" L2 Loss: {l2_loss_avg:.2f}"
                )

        # Evaluation loop
        for batch_num, batch in enumerate(val_dataset):
            boards, policies, masks, values = batch
            evaluation_step(model, loss_function, boards, policies, masks, values)
            val_loss_metric.update_state(loss_function)

            # Print update every 'update_freq' batches
            if (batch_num + 1) % update_freq == 0:
                (
                    total_loss_avg,
                    policy_loss_avg,
                    value_loss_avg,
                    l2_loss_avg,
                ) = val_loss_metric.result()
                print(
                    f"Epoch {epoch}, Batch {batch_num+1},"
                    f" Training Loss: {total_loss_avg:.2f},"
                    f" Policy Loss: {policy_loss_avg:.2f},"
                    f" Value Loss: {value_loss_avg:.2f},"
                    f" L2 Loss: {l2_loss_avg:.2f}"
                )

        # End of epoch
        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()
        print(f"Epoch {epoch} - train loss: {train_loss}, val loss: {val_loss}")

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

    # Load the best model at the end of training
    checkpoint.restore(manager.latest_checkpoint)

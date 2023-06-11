#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:45:25 2023

@author: chris
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.joinpath("src")))

import time

import numpy as np
import tensorflow as tf

from mapping import BoardMapper, PolicyMapper
from neural_network.architecture import AlphaZeroNetwork
from neural_network.preprocessing import prepare_dataset
from utilities import DataHandler

# %%
boards, policies, values = DataHandler().load()
board_mapper = BoardMapper()
policy_mapper = PolicyMapper()
train_data, val_data = prepare_dataset(boards, policies, values)

# %%
model = AlphaZeroNetwork(256, 19)

#%%

# def train(model, epochs, train_dataset, val_dataset, optimizer, 
#           loss_function, checkpoint_dir):
#     """
#     Training loop for the AlphaZeroNetwork with checkpoints.

#     Parameters
#     ----------
#     model : AlphaZeroNetwork
#         The AlphaZeroNetwork model to be trained.
#     epochs : int
#         Number of epochs to train for.
#     train_dataset : tf.data.Dataset
#         The training dataset.
#     val_dataset : tf.data.Dataset
#         The validation dataset.
#     optimizer : tf.keras.optimizers.Optimizer
#         The optimizer to use.
#     loss_function : tf.keras.losses.Loss
#         The loss function to use.
#     checkpoint_dir : str
#         Directory where to save checkpoints.

#     """
#     train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
#     val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

#     # Set up checkpoint objects
#     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
#     manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

#     for epoch in range(epochs):
#         print(f"Start of epoch {epoch}")

#         # Training loop
#         for batch in train_dataset:
#             with tf.GradientTape() as tape:
#                 # Forward pass
#                 policy_logits, value = model(batch['board'])
#                 # Compute loss
#                 policy_loss = loss_function(batch['policy'], policy_logits)
#                 value_loss = loss_function(batch['value'], value)
#                 total_loss = policy_loss + value_loss
#             # Compute gradients and update weights
#             gradients = tape.gradient(total_loss, model.trainable_weights)
#             optimizer.apply_gradients(zip(gradients, model.trainable_weights))

#             # Update training metric
#             train_loss_metric.update_state(total_loss)

#         # Evaluation loop
#         for batch in val_dataset:
#             # Forward pass
#             policy_logits, value = model(batch['board'])
#             # Compute loss
#             policy_loss = loss_function(batch['policy'], policy_logits)
#             value_loss = loss_function(batch['value'], value)
#             total_loss = policy_loss + value_loss

#             # Update validation metric
#             val_loss_metric.update_state(total_loss)

#         # End of epoch
#         train_loss = train_loss_metric.result()
#         val_loss = val_loss_metric.result()
#         print(f"Epoch {epoch} - train loss: {train_loss}, val loss: {val_loss}")

#         # Save checkpoint
#         manager.save()

#         # Reset metrics for the next epoch
#         train_loss_metric.reset_states()
#         val_loss_metric.reset_states()


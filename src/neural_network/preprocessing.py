#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:25:14 2023

@author: chris
"""
from typing import Callable, Generator

import tensorflow as tf

from mapping import BoardMapper, PolicyMapper


def create_generator(
    boards: list,
    policies: list,
    values: list,
) -> Callable:
    """
    Create a generator for a TensorFlow dataset.

    Parameters
    ----------
    boards : list
        List of board states (FENs).
    policies : list
        List of policy dicts.
    values : list
        List of state values.

    Returns
    -------
    Callable
        The created generator function.

    """
    board_mapper = BoardMapper()
    policy_mapper = PolicyMapper()

    def generator() -> Generator:
        for board, policy, value in zip(boards, policies, values):
            board_tensor = tf.convert_to_tensor(
                board_mapper.create(board), dtype=tf.float32
            )
            policy_array, mask_array = policy_mapper.create(policy)
            policy_tensor = tf.convert_to_tensor(policy_array, dtype=tf.float32)
            mask_tensor = tf.convert_to_tensor(mask_array, dtype=tf.bool)
            value_tensor = tf.convert_to_tensor([value], dtype=tf.int32)
            yield (board_tensor, policy_tensor, mask_tensor, value_tensor)

    return generator


def name_map(*args: tuple) -> dict[str, tf.Tensor]:
    """
    Map tensor tuple to dictionary.

    Parameters
    ----------
    *args : tuple
        Tuple of tensors.

    Returns
    -------
    dict[str, tf.Tensor]
        Dictionary corresponding to tuple.

    """
    return {
        "boards": args[0],
        "policies": args[1],
        "masks": args[2],
        "values": args[3],
    }


def prepare_dataset(
    boards: list,
    policies: list,
    values: list,
    buffer_size: int = 10000,
    batch_size: int = 32,
    train_val_split: float = 0.8,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare a TensorFlow dataset.

    Parameters
    ----------
    boards : list
        List of board states (FENs).
    policies : list
        List of policy dicts.
    values : list
        List of state values.
    board_mapper : Callable
        Function to map boards FENs to neural network input tensors.
    policy_mapper : Callable
        Function to map policy dictionaries to tensors.
    buffer_size : int, optional
        Buffer size for shuffling. The default is 10000.
    batch_size : int, optional
        Batch size. The default is 32.
    train_val_split : float, optional
        Proportion of the dataset to include in the training set, remainder is
        added to validation set. The default is 0.8.


    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset]
        The prepared TensorFlow training and validation datasets.

    """
    dataset = tf.data.Dataset.from_generator(
        create_generator(boards, policies, values),
        output_signature=(
            tf.TensorSpec(
                shape=(8, 8, 20),
                dtype=tf.float32,
            ),  # boards
            tf.TensorSpec(shape=(1968,), dtype=tf.float32),  # policies
            tf.TensorSpec(shape=(1968,), dtype=tf.bool),  # masks
            tf.TensorSpec(shape=(1,), dtype=tf.int32),  # values
        ),
    )

    dataset = dataset.map(name_map)

    # Shuffle dataset
    dataset = dataset.shuffle(buffer_size)

    # Calculate train and test sizes
    data_size = len(boards)
    train_size = int(data_size * train_val_split)
    val_size = data_size - train_size

    # Create train and test datasets
    train_set = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_set = dataset.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_set, validation_set

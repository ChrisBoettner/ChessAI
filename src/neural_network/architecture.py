#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:43:53 2023

@author: chris
"""

from typing import Tuple

import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    """
    A convolutional block composed of a convolution layer, a batch normalization,
    and a ReLU activation.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int] = (3, 3),
        apply_relu: bool = True,
    ) -> None:
        """
        Initialize ConvBlock layer.

        Parameters
        ----------
        filters : int
            The number of filters the convolutional layers will learn from the data.
        kernel_size : tuple of int, optional
            Specifies the height and width of the 2D convolution window.
        apply_relu : bool, optional
            Determines whether to apply ReLU activation function. The default is True.
        """
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.apply_relu = apply_relu
        if apply_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the ConvBlock layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        x = self.conv(inputs)
        x = self.batch_norm(x)
        if self.apply_relu:
            x = self.relu(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    """
    A residual block composed of two ConvBlocks with a skip connection.
    """

    def __init__(self, filters: int) -> None:
        """
        Initialize ResidualBlock layer.

        Parameters
        ----------
        filters : int
            The number of filters the convolutional layers will learn from the data.
        """
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(filters)
        self.conv_block2 = ConvBlock(filters, apply_relu=False)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the ResidualBlock layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.relu(x + inputs)  # skip connection
        return x


class PolicyHead(tf.keras.layers.Layer):
    """
    Policy head of the AlphaZero network architecture. The final layer is a Dense layer
    with softmax activation.
    """

    def __init__(self, filters: int = 2, classes: int = 1968) -> None:
        """
        Initialize PolicyHead layer.

        Parameters
        ----------
        filters : int, optional
            The number of filters the convolutional layers will learn from the data. The
            default is 2.
        classes : int, optional
            The number of possible actions. The chess UCI allows 1968 possible moves.
        """
        super(PolicyHead, self).__init__()
        self.conv_block = ConvBlock(filters, kernel_size=(1, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(classes, activation=None, name="policy")

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        """
        Forward pass through the PolicyHead layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        mask : tf.Tensor, optional
            A tensor containing 1s at positions corresponding to legal moves,
            and 0s elsewhere.

        Returns
        -------
        tf.Tensor
            Output tensor (policy distribution over the actions).
        """
        x = self.conv_block(inputs)
        x = self.flatten(x)
        x = self.dense(x)

        if mask is not None:
            # Replace illegal moves (mask=0) values with a large negative value so
            # softmax makes it approximately zero
            x = tf.where(mask, x, -1e15)
            x = tf.nn.softmax(x, axis=-1)  # Apply softmax on each individual example
        return x


class ValueHead(tf.keras.layers.Layer):
    """
    Value head of the AlphaZero network architecture. The final layer is a Dense layer
    with tanh activation.
    """

    def __init__(self, filters: int = 1, dense_neurons: int = 256) -> None:
        """
        Initialize ValueHead layer.

        Parameters
        ----------
        filters : int, optional
            The number of filters the convolutional layer will learn from the data. The
            default is 1.
        dense_neurons : int, optional
            The number of neurons in the dense layer. The default is 256.
        """
        super(ValueHead, self).__init__()
        self.conv_block = ConvBlock(filters, kernel_size=(1, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(dense_neurons, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="tanh", name="value")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the ValueHead layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor (expected outcome of the game).
        """
        x = self.conv_block(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class AlphaZeroNetwork(tf.keras.Model):
    """
    AlphaZero neural network architecture.
    """

    def __init__(self, filters: int, res_blocks: int) -> None:
        """
        Initialize AlphaZeroNetwork.

        Parameters
        ----------
        filters : int
            The number of filters the convolutional layers will learn from the data.
        res_blocks : int
            The number of residual blocks.
        """
        super(AlphaZeroNetwork, self).__init__()
        self.conv_block = ConvBlock(filters)
        self.res_blocks = [ResidualBlock(filters) for _ in range(res_blocks)]
        self.policy_head = PolicyHead(filters)
        self.value_head = ValueHead(filters)

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through the AlphaZeroNetwork.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        mask : tf.Tensor, optional
            A tensor containing 1s at positions corresponding to legal moves,
            and 0s elsewhere.

        Returns
        -------
        tuple of tf.Tensor
            Output tensors (policy distribution and expected outcome).
        """
        x = self.conv_block(inputs)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy_outputs = self.policy_head(x, mask)
        value_outputs = self.value_head(x)
        return policy_outputs, value_outputs

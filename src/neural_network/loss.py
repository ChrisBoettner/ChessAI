#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:12:01 2023

@author: chris
"""
import tensorflow as tf


class AlphaZeroLoss(tf.keras.losses.Loss):
    """
    AlphaZero loss function with optional L2 regularization.

    """

    def __init__(
        self, model: tf.keras.Model, l2_reg: float = 0.0, name: str = "alpha_zero_loss"
    ) -> None:
        """
        Initialize AlphaZero loss function.

        Parameters
        ----------
        model : tf.keras.Model
            The model to apply the loss function to.
        l2_reg : float, optional
            L2 regularization factor. Default is 0.0, which means no regularization.
        name : str, optional
            Name of the loss function. The default is 'alpha_zero_loss'.

        """
        super().__init__(name=name)
        self.l2_reg = l2_reg
        self.model = model

    def call(
        self, y_true: tuple[tf.Tensor, tf.Tensor], y_pred: tuple[tf.Tensor, tf.Tensor]
    ) -> float:
        """
        Compute the AlphaZero loss.

        Parameters
        ----------
        y_true : tuple[tf.Tensor, tf.Tensor]
            Tuple containing the true policy and the true value.
        y_pred : tuple[tf.Tensor, tf.Tensor]
            Tuple containing the predicted policy and the predicted value.

        Returns
        -------
        float
            Computed total AlphaZero loss with optional L2 regularization.

        """
        policy_true, value_true = y_true
        policy_pred, value_pred = y_pred

        # Cross-entropy loss for policy
        policy_loss = tf.keras.losses.categorical_crossentropy(policy_true, policy_pred)

        # Mean squared error loss for value
        value_loss = tf.keras.losses.mean_squared_error(value_true, value_pred)

        # L2 regularization
        l2_loss = self.l2_reg * sum(
            [tf.nn.l2_loss(v) for v in self.model.trainable_variables]
        )

        # calculate total loss
        total_loss = policy_loss + value_loss + l2_loss

        # save losses to attributes for logging
        self.policy_loss = tf.reduce_mean(policy_loss)
        self.value_loss = tf.reduce_mean(value_loss)
        self.l2_loss = tf.reduce_mean(l2_loss)
        self.total_loss = tf.reduce_mean(total_loss)

        return total_loss

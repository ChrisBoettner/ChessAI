#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:22:15 2023

@author: chris
"""
from typing import Any

import tensorflow as tf


class AlphaZeroLossMetric(tf.keras.metrics.Metric):
    """
    Custom Keras metric for tracking AlphaZero loss components (total, policy,
    value, and L2) over multiple batches.
    """

    def __init__(self, name: str = "alpha_zero_loss_metric", **kwargs: Any) -> None:
        """
        Initializes the AlphaZeroLossMetric metric.

        Parameters
        ----------
        name : str, optional
            Name of the metric.
        **kwargs
            Additional keyword arguments.
        """
        super(AlphaZeroLossMetric, self).__init__(name=name, **kwargs)
        self.total_loss_tracker = self.add_weight(
            name="total_loss", initializer="zeros"
        )
        self.policy_loss_tracker = self.add_weight(
            name="policy_loss", initializer="zeros"
        )
        self.value_loss_tracker = self.add_weight(
            name="value_loss", initializer="zeros"
        )
        self.l2_loss_tracker = self.add_weight(name="l2_loss", initializer="zeros")
        self.num_batches = self.add_weight(name="num_batches", initializer="zeros")

    def update_state(
        self, loss_function: tf.keras.losses.Loss, *args: Any, **kwargs: Any
    ) -> None:
        """
        Updates the state of the metric.

        Parameters
        ----------
        loss_function : tf.keras.losses.Loss
            Instance of a custom loss function that contains `total_loss`,
            `policy_loss`, `value_loss` and `l2_loss` attributes.
        *args, **kwargs
            Additional positional and keyword arguments.
        """
        self.total_loss_tracker.assign_add(loss_function.total_loss)
        self.policy_loss_tracker.assign_add(loss_function.policy_loss)
        self.value_loss_tracker.assign_add(loss_function.value_loss)
        self.l2_loss_tracker.assign_add(loss_function.l2_loss)
        self.num_batches.assign_add(1)

    def result(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Computes and returns the average losses (total, policy, value, and L2) over
        all tracked batches.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            Average total loss, average policy loss, average value loss and average
            L2 loss.
        """
        total_loss_avg = tf.math.divide_no_nan(
            self.total_loss_tracker, self.num_batches
        )
        policy_loss_avg = tf.math.divide_no_nan(
            self.policy_loss_tracker, self.num_batches
        )
        value_loss_avg = tf.math.divide_no_nan(
            self.value_loss_tracker, self.num_batches
        )
        l2_loss_avg = tf.math.divide_no_nan(self.l2_loss_tracker, self.num_batches)
        return total_loss_avg, policy_loss_avg, value_loss_avg, l2_loss_avg

    def reset_states(self) -> None:
        """
        Resets the state of the metric. This method is called at the start of each
        epoch.
        """
        self.total_loss_tracker.assign(0.0)
        self.policy_loss_tracker.assign(0.0)
        self.value_loss_tracker.assign(0.0)
        self.l2_loss_tracker.assign(0.0)
        self.num_batches.assign(0.0)

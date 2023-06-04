#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:38:47 2023

@author: chris
"""
import pathlib
from typing import Any

from ruamel.yaml import YAML


def flatten_list(nested_list: list[list]) -> list[Any]:
    """
    Flatten sublists in list.

    Parameters
    ----------
    nested_list : list[list]
        Nested list.

    Returns
    -------
    list[Any]
        Flattened list.

    """
    return [item for sublist in nested_list for item in sublist]


def load_hyperparameter(obj: object, hyperparameter_tag: str) -> None:
    """
    Read values from YAML file and write to attributes of object.

    """
    hyperparameter_file = YAML().load(pathlib.Path("hyperparameter.yaml"))
    hyperparameter = dict(hyperparameter_file[hyperparameter_tag])
    for key, value in hyperparameter.items():
        setattr(obj, key, value)

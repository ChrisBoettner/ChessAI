#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:38:47 2023

@author: chris
"""
import json
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
    hyperparameter_file = YAML().load(pathlib.Path("data/hyperparameter.yaml"))
    hyperparameter = dict(hyperparameter_file[hyperparameter_tag])
    for key, value in hyperparameter.items():
        setattr(obj, key, value)


class DataHandler:
    """
    Class to save and load game data.

    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def save(
        boards: list[str],
        policies: list[dict],
        values: list[int],
        filename: str = "game_data",
    ) -> None:
        """
        Save game data to JSON file.

        Parameters
        ----------
        boards : list[str]
            FENs of board state.
        policies : list[dict]
            Policies at state, of form {move uci: policy value}.
        values : list[int]
            Value associated with state (-1 if lose, 0 if draw, 1 if win).
        filename : str, optional
            Name of file data is saved to. The default is "game_data".

        """
        # Ensure all lists are of the same length
        assert len(boards) == len(values) == len(policies)

        # Combine your data into a list of dictionaries that can be serialized to JSON
        data = [
            {
                "boards": b,
                "values": int(v),
                "policies": {k: float("{:.2e}".format(v)) for k, v in p.items()},
            }
            for b, v, p in zip(boards, values, policies)
        ]

        # Save data to a JSON file
        with open(f"data/{filename}.json", "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filename: str = "game_data") -> tuple[list, list, list]:
        """
        Load game data from file.

        Parameters
        ----------
        filename : str, optional
            Name of file data is laoded from. The default is "game_data".

        Returns
        -------
        tuple[list,list,list]
            boards, policies, values.

        """

        with open(f"data/{filename}.json", "r") as f:
            data = json.load(f)

        # Split the loaded data back into separate lists
        boards = [item["boards"] for item in data]
        values = [item["values"] for item in data]
        policies = [item["policies"] for item in data]

        return boards, policies, values


def read_file(filename: str) -> list:
    """
    Read txt file to list.

    Parameters
    ----------
    filename : str
        Path and name of file.

    Returns
    -------
    list
        File contents.

    """
    with open(filename, "r") as file:
        lines = [line.rstrip("\n") for line in file]
    return lines

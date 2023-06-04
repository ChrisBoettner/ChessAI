#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:54:28 2023

@author: chris
"""
from typing import Dict

import numpy as np
from numpy.typing import NDArray


class InputMapper:
    """
    Mapping class to convert FENs to CNN input states.
    """

    def __init__(self) -> None:
        """
        Initialize mapper, create mapping dicts.

        """
        self.player_mapping: Dict[str, int] = {"w": 0, "b": 1}

        self.piece_mapping_white: Dict[str, int] = {
            "P": 0,
            "R": 1,
            "N": 2,
            "B": 3,
            "Q": 4,
            "K": 5,
            "p": 6,
            "r": 7,
            "n": 8,
            "b": 9,
            "q": 10,
            "k": 11,
        }

        self.piece_mapping_black: Dict[str, int] = {
            "p": 0,
            "r": 1,
            "n": 2,
            "b": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "R": 7,
            "N": 8,
            "B": 9,
            "Q": 10,
            "K": 11,
        }

        self.castling_mapping_white: Dict[str, int] = {
            "K": 13,
            "Q": 14,
            "k": 15,
            "q": 16,
        }
        self.castling_mapping_black: Dict[str, int] = {
            "k": 13,
            "q": 14,
            "K": 15,
            "Q": 16,
        }

        self.enpassant_mapping: Dict[str, int] = {
            "-": 0,
            "a3": 1,
            "a6": 2,
            "b3": 3,
            "b6": 4,
            "c3": 5,
            "c6": 6,
            "d3": 7,
            "d6": 8,
            "e3": 9,
            "e6": 10,
            "f3": 11,
            "f6": 12,
            "g3": 13,
            "g6": 14,
            "h3": 15,
            "h6": 16,
        }

    def fen_to_board(self, split_fen: list[str], array: NDArray) -> NDArray:
        """
        Fill array with board states from FEN.

        Parameters
        ----------
        split_fen : str
            FEN split by whitespace.
        array : NDArray
            Empty array to be filled.

        Returns
        -------
        NDArray
            CNN input array created from FEN, filled with board states.

        """
        # choose piece mapping (board orientation) based on player colour
        piece_mapping = (
            self.piece_mapping_white
            if split_fen[1] == "w"
            else self.piece_mapping_black
        )

        # fill board states
        for i, row in enumerate(split_fen[0].split("/")):
            loc = 0
            for char in row:
                if char in "12345678":
                    loc += int(char)
                else:
                    if split_fen[1] == "w":
                        array[i, loc, piece_mapping[char]] = 1
                    else:  # rotate by 90deg if player colour is black
                        array[7 - i, loc, piece_mapping[char]] = 1
                    loc += 1
        return array

    def fen_to_additional(self, split_fen: list[str], array: NDArray) -> NDArray:
        """
        Fill array with additional information from FEN.

        Parameters
        ----------
        split_fen : str
            FEN split by whitespace.
        array : NDArray
            Empty array to be filled.

        Returns
        -------
        NDArray
            CNN input array created from FEN, filled with additional information.

        """
        # choose castling mapping based on player colour
        castling_mapping = (
            self.castling_mapping_white
            if split_fen[1] == "w"
            else self.castling_mapping_black
        )

        # fill channels with additional information
        for i, field in enumerate(split_fen[1:]):
            match i:
                case 0:  # player colour
                    array[:, :, 12] = self.player_mapping[field]
                case 1:  # castling rights
                    if field == "-":
                        pass
                    else:
                        for char in field:
                            array[:, :, castling_mapping[char]] = 1
                case 2:  # en passant availability
                    array[:, :, 17] = self.enpassant_mapping[field]
                case 3:  # half-move clock
                    array[:, :, 18] = int(field)
                case 4:  # fullmove counter
                    array[:, :, 19] = int(field)
                case _:
                    raise ValueError("FEN not valid.")

        return array

    def create_input(self, fen: str) -> NDArray:
        """
        Convert FEN to input array.

        Parameters
        ----------
        fen : str
            FEN of game state.

        Returns
        -------
        NDArray
            CNN input array created from FEN.

        """
        input_array = np.zeros((8, 8, 20), dtype=np.int16)

        input_array = self.fen_to_board(fen.split(" "), input_array)
        input_array = self.fen_to_additional(fen.split(" "), input_array)
        return input_array


if __name__ == "__main__":
    from training_game import create_training_data

    boards, policies, values = create_training_data(10)
    im = InputMapper()

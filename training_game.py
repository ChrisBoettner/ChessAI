#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:14:57 2023

@author: chris
"""
import multiprocessing
import warnings
from typing import Any, Optional

import chess.engine
import numpy as np
from chess.engine import InfoDict
from scipy.special import softmax

from mcts import Game, Node
from utilities import flatten_list, load_hyperparameter


class TrainingGame(Game):
    """
    Class to create training games using stockfish.

    """

    def __init__(
        self,
        root_state: Optional[str] = None,
        engine: str = "stockfish",
        engine_skill: int | str | tuple = 20,
    ) -> None:
        """
        Initialize training game.

        Parameters
        ----------
        root_state : str, optional
            FEN of initital state. The default is None, which corresponds to a
            starting board.
        engine : str, optional
            Name of the engine used. The default is "stockfish".
        engine_skill : int, optional
            Skill level of the engine. Can be a number or "random". If one value is
            given, use for both players. If tuple is given, use each entry for one of
            the players. The default is 20.

        """
        self.root_state = root_state
        self.reset()

        self.TEMPERATURE = 0  # filled with load_hyperparameter
        self.TIME_LIMIT = 0  # filled with load_hyperparameter
        self.MATE_SCORE = 0  # filled with load_hyperparameter
        load_hyperparameter(self, "TRAINING")

        if engine == "stockfish":
            self.engine = chess.engine.SimpleEngine.popen_uci(
                r"stockfish/stockfish_15.1_x64_bmi2"
            )
            self.set_skill_level(engine_skill)

    def make_move(self, node: Node, action: str) -> Node:
        """
        Make move by taking node, pushing action on node.board, and returning same node.

        Parameters
        ----------
        node : Node
            The node containing the board state.
        action : str
            UCI of move to make.

        Returns
        -------
        Node
            The node containing the board state, now advanced by action.

        """
        node.make_move(action)
        return node

    def get_policy(
        self,
        node: Node,
    ) -> dict[str, float]:
        """
        Get policy by letting stockfish evaluate the next move, and using softmax on
        the stockfish values.

        Parameters
        ----------
        node : Node
            The node containing the board state.

        Returns
        -------
        dict[str, float]
            Dict of policy, of form {move uci: policy value}.

        """
        infos = self.engine.analyse(
            node.board,
            chess.engine.Limit(time=self.TIME_LIMIT),
            multipv=500,  # maximum number of possible moves considered
            options={
                "Skill Level": self.engine_skill[self.move_counter % 2]
            },  # probably useless
        )
        return self.calculate_policy(infos, self.TEMPERATURE, self.MATE_SCORE)

    @staticmethod
    def calculate_policy(
        infos: list[InfoDict],
        temperature: float,
        mate_score: int,
    ) -> dict[str, float]:
        """
        Calculate score from InfoDicts returned by stockfish and calculating softmax on
        those

        Parameters
        ----------
        infos : list[Info]
            InfoDicts for every move considered by stockfish, containing all
            information on the move, including scores.
        temperature : float
            Temperature parameter that balances exploration and exploitation. Higher
            values lead to more random moves.
        mate_score : int
            Score associated with board states that lead to a check mate.

        Returns
        -------
        dict[str, float]
            Dict of policy, of form {move uci: policy value}.

        """
        scores = [info["score"].relative.score(mate_score=mate_score) for info in infos]

        policy = softmax(scores) ** (1 / temperature)
        policy /= np.sum(policy)

        scores_dict = {info["pv"][0].uci(): pol for info, pol in zip(infos, policy)}
        return scores_dict

    def set_skill_level(self, engine_skill: int | str | tuple) -> None:
        """
        Set skill level of stockfish engine.

        Parameters
        ----------
        engine_skill : int, optional
            Skill level of the engine. Can be a number or "random". If one value is
            given, use for both players. If tuple is given, use each entry for one of
            the players. The default is 20.

        """
        if engine_skill != 20:
            warnings.warn(
                "CAUTION: ENGINE_SKILL MIGHT NOT ACTUALLY HAVE AN EFFECT BASED ON \
                 HOW MOVES ARE SELECTED. TO CHANGE SKILL LEVEL, TIME_LIMIT AND \
                 TEMPERATURE MIGHT BE MORE APPROPRIATE PARAMETER.",
                stacklevel=2,
            )

        match engine_skill:
            case "random":
                skill_level = (self.random_skill_level(), self.random_skill_level())
            case int():
                skill_level = (engine_skill, engine_skill)
            case (int(), int()):
                pass
            case ("random", int()):
                skill_level = (self.random_skill_level(), engine_skill[1])
            case ("random", int()):
                skill_level = (engine_skill[0], self.random_skill_level())
            case ("random", "random"):
                skill_level = (
                    self.random_skill_level(),
                    self.random_skill_level(),
                )
            case _:
                raise ValueError("engine_skill input not understood.")
        self.engine_skill = skill_level

    @staticmethod
    def random_skill_level(power: float = 0.2) -> int:
        """
        Choose random skill level between 0 and 20. Highest probability is skill level
        10.5, decaying symmetrically outward.

        Parameters
        ----------
        power : float, optional
            Power of weights, higher values lead to stronger decay of probability.
            The default is 0.2.

        Returns
        -------
        float
            Skill level, random number between 0 and 20.

        """
        levels = np.arange(1, 21)
        weights = np.abs((levels - np.mean(levels))) ** (-power)
        return np.random.choice(levels, p=weights / sum(weights))


def get_game_data(i: Optional[int] = None) -> list:
    """
    Create TrainingGame object, play game, quit engine and return results. Mainly used
    for parallelizing.

    Parameters
    ----------
    i : int, optional
        Primarely needed for multiprocessing. The default is None.

    Returns
    -------
    list[list]
        Game results, containing game states, policies and target values.

    """
    game = TrainingGame()
    game_states, policies, target_values = game.play()
    game.engine.quit()
    del game
    return [game_states, policies, target_values]


def process_game_data(
    game_data: list[list],
    flatten: bool = True,
) -> list[list]:
    """
    Process game data returned by get_game_data. Reshuffle entries so it resturns
    list of three lists: First one contains board states, second contains policy dicts,
    third one contains game states.

    Parameters
    ----------
    game_data : list[list,list,list]
        list of game data results from get_game_Data.
    flatten : TYPE, optional
        If True, inner lists are flattened. The default is True.

    Returns
    -------
    list[list]
        Processed game data.

    """
    game_data = [list(i) for i in zip(*game_data)]
    if flatten:
        game_data = [flatten_list(i) for i in game_data]
    return game_data


def create_training_data(
    num: int = 30, parallel: bool = True, **kwargs: Any
) -> list[list]:
    """
    Play a number of training games and return processed results

    Parameters
    ----------
    num : int, optional
        Number of games. The default is 30.
    parallel : bool, optional
        If True, parallelize using multiprocessing. The default is True.
    **kwargs : Any
        Extra parameter passed to process_game_data.

    Returns
    -------
    list
        Processed game data.

    """
    if parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        game_data = pool.map(get_game_data, range(num))
    else:
        game_data = [get_game_data(i) for i in range(num)]
    return process_game_data(game_data, **kwargs)

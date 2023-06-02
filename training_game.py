#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:14:57 2023

@author: chris
"""
import multiprocessing

import chess.engine
import numpy as np
from chess import Move
from scipy.special import softmax

import warnings

from utilities import flatten_list

from mcts import Game


class TrainingGame(Game):
    """
    Class to create training games using stockfish.
    """

    def __init__(
        self,
        root_state: str = None,
        engine: str = "stockfish",
        engine_skill: int | str | tuple = 20,
    ) -> None:
        """
        Initilize Training Game.

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

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.root_state = root_state
        self.reset()

        if engine == "stockfish":
            self.engine = chess.engine.SimpleEngine.popen_uci(
                r"stockfish/stockfish_15.1_x64_bmi2"
            )
            self.set_skill_level(engine_skill)

    def make_move(self, node, action):
        node.board.push(Move.from_uci(action))
        return node

    def get_policy(self, node, time_limit=0.01, temperature=1, mate_score=int(1e5)):
        infos = self.engine.analyse(
            node.board,
            chess.engine.Limit(time=time_limit),
            multipv=500,
            options={"Skill Level": self.engine_skill[self.move_counter % 2]},
        )
        return self.calculate_score(infos, temperature, mate_score)

    @staticmethod
    def calculate_score(infos, temperature, mate_score):
        scores = [info["score"].relative.score(mate_score=mate_score) for info in infos]
        scores = softmax(scores) ** (1 / temperature)
        scores = scores / np.sum(scores)

        scores = {info["pv"][0].uci(): score for info, score in zip(infos, scores)}
        return scores

    def set_skill_level(self, engine_skill) -> None:
        if engine_skill != 20:
            warnings.warn(
                "CAUTION: ENGINE_SKILL MIGHT NOT ACTUALLY HAVE AN EFFECT BASED ON \
                 HOW MOVES ARE SELECTED. TO CHANGE SKILL LEVEL, TIME_LIMIT AND \
                 TEMPERATURE MIGHT BE MORE APPROPRIATE PARAMETER.",
                stacklevel=2,
            )

        match engine_skill:
            case "random":
                engine_skill = self.random_skill_level()
                engine_skill = (engine_skill, engine_skill)
            case int():
                engine_skill = (engine_skill, engine_skill)
            case (int(), int()):
                pass
            case ("random", int()):
                engine_skill = (self.random_skill_level(), engine_skill[1])
            case ("random", int()):
                engine_skill = (engine_skill[0], self.random_skill_level())
            case ("random", "random"):
                engine_skill = (
                    self.random_skill_level(),
                    self.random_skill_level(),
                )
            case _:
                raise ValueError("engine_skill input not understood.")

        self.engine_skill = engine_skill

    @staticmethod
    def random_skill_level(power: float = 0.2) -> float:
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


def get_game_data(i=None):
    game = TrainingGame()
    a, b, c = game.play()
    game.engine.quit()
    del game
    return [a, b, c]


def process_game_data(game_data, flatten=True):
    game_data = [list(i) for i in zip(*game_data)]
    if flatten:
        game_data = [flatten_list(i) for i in game_data]
    return game_data


def create_training_data(num=30, parallel=True, **kwargs):
    if parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        game_data = pool.map(get_game_data, range(num))
    else:
        game_data = [get_game_data(i) for i in range(num)]
    return process_game_data(game_data, **kwargs)
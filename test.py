#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:24:09 2023

@author: chris
"""

from typing import Optional

import chess
import chess.engine


class Game(object):
    def __init__(
        self,
        engine: str = "stockfish",
        player_colour: str = "white",
        engine_limit: float = 0.1,
        engine_skill_level: int = 10,
    ) -> None:
        # load engine
        if engine == "stockfish":
            self.engine = chess.engine.SimpleEngine.popen_uci(
                r"stockfish/stockfish"
            )
        # set attributes
        if player_colour.casefold() in ["white", "black"]:
            self.player_colour = player_colour
        else:
            raise ValueError("player_colour must be white or black.")
        self.board = chess.Board()
        self.engine_limit = engine_limit
        # configure engine
        self.engine.configure({"Skill Level": engine_skill_level})

    def start_game(self) -> None:
        self.reset_game()
        self._game()

    def continue_game(self) -> None:
        raise NotImplementedError(
            "Does not quite work yet, since we do not " "save whos turn it is."
        )
        self._game()

    def reset_game(self) -> None:
        self.board.reset()

    def _game(self) -> None:
        # player order
        turns = (
            [self._player_turn, self._engine_turn]
            if self.player_colour == "white"
            else [self._engine_turn, self._player_turn]
        )

        # play
        while not self.board.is_game_over():
            for turn in turns:
                move = turn()
                if isinstance(move, chess.Move):
                    self.board.push(move)
                    if self.board.is_game_over():
                        break
                else:
                    print("Qutting game.")
                    return

        # print outcome
        outcome = self.board.outcome()
        assert outcome is not None

        if outcome.winner is None:
            print(f"Winner: Draw - {outcome.termination.name}")
        elif outcome.winner is False:
            print("Winner: Black")
        else:
            print("Winner: White")

    def _engine_turn(self) -> Optional[chess.Move]:
        engine_move = self.engine.play(
            self.board, chess.engine.Limit(time=self.engine_limit)
        ).move
        return engine_move

    def _player_turn(self) -> chess.Move | None:
        # list legal moves
        moves = list(self.board.legal_moves)
        moves_dict = {str((1 + n)): moves[n] for n in range(len(moves))}

        # print moves
        for key, m in moves_dict.items():
            key = key if len(str(key)) == 2 else (" " + str(key))
            move_name = m.uci()[:2] + " -> " + m.uci()[2:]
            print(f"{key}    :   {move_name}")
        print("quit  :       quit\n")
        print(self.board)
        print()

        # input prompt prompt
        while True:
            player_move = input("Choose your move: ")
            if player_move in list(moves_dict.keys()):
                return moves_dict[player_move]
            elif player_move in ["q", "quit"]:
                break
            else:
                print(
                    "  Input must be correspond to a move value or quit." " Try again."
                )
        return None


if __name__ == "__main__":
    from scipy.special import softmax

    g = Game()
    board = g.board
    engine = g.engine

    t = []
    for el in board.legal_moves:
        info = engine.analyse(board, chess.engine.Limit(time=0.005), root_moves=[el])
        t.append(info["score"].relative.score())
        sm = softmax(t)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:24:02 2023

@author: chris
"""
from __future__ import (
    annotations,  # used so that Node can references itself in type hint
)

from typing import Any, Optional

import numpy as np
from chess import Board, Move
from numpy.typing import NDArray

from utilities import load_hyperparameter


class Node:
    """
    Node of game tree.

    """

    def __init__(
        self,
        state: Optional[str] = None,
        parent: Optional[Node] = None,
        parent_policy_estimate: float = 1,
    ) -> None:
        """
        Initialize node.

        Parameters
        ----------
        state : str
            Game state of Node (FEN).
        parent : Node, optional
            Parent node in tree. The default is None, corresponding to the root node.
        parent_policy_estimate : float, optional
            The policy value for transition from parent to this node estimated by neural
            network. The default is 1.

        """
        self.C = 0  # filled with load_hyperparameter
        load_hyperparameter(self, "NODE")

        self.state = state

        # if state is given, create corresponding board
        if self.state is not None:
            self.board = Board(self.state)
            self.game_over = self.board.is_game_over()

        # parents and children
        self.parent = parent
        self.children: dict[str, Node] = {}
        self.legal_actions: list[str] = []

        # when initialized, Node is leaf
        self.is_leaf = True

        # values estimated by neural network
        self.nn_value_estimate: Optional[float] = None
        self.nn_policy_estimate: Optional[dict] = None

        # values connected to policy and PUCT calculation
        self.visits = 0
        self.value = 0.0
        self.parent_policy_estimate = (
            parent_policy_estimate  # policy value estimated by parent for this node
        )

        self.calculate_puct()

    def fill(self, action: str) -> None:
        """
        Fill empty node with chosen action.

        Parameters
        ----------
        action : str
            Action chosen to get from parent node to this node.

        """
        if self.state is None:
            self.make_parent_board()
            self.make_move(action)
            self.set_state()
        else:
            pass

    def make_parent_board(self) -> None:
        """
        Initialize board as parent board. make_move will turn it into board associated
        with node.

        """
        if self.parent is not None:
            self.board = Board(self.parent.state)
        else:
            raise AttributeError("Node parent is None. Can't create board.")

    def make_move(self, action: str) -> None:
        """
        Advance board state based on chosen action.

        Parameters
        ----------
        action : str
            UCI of action.

        """
        self.board.push(Move.from_uci(action))
        self.check_game_over()

    def set_state(self) -> None:
        """
        Set state to current board fen.

        """
        self.state = self.board.fen()

    def check_game_over(self) -> None:
        """
        Check if game is over.

        """
        self.game_over = self.board.is_game_over()
        if self.game_over:
            outcome = self.board.outcome()
            assert outcome is not None
            self.outcome = outcome

    def expand(self) -> None:
        """
        Create children nodes, set legal_actions, set is_leaf = False.

        """
        policy_estimate = self.get_nn_policy_estimate()

        for move in self.board.legal_moves:
            self.children[move.uci()] = Node(
                None, parent=self, parent_policy_estimate=policy_estimate[move.uci()]
            )

        self.legal_actions = list(self.children.keys())
        self.is_leaf = False

    def get_nn_value_estimate(self) -> float:
        """
        Get value estimate for this node from neural network.

        Returns
        -------
        float
            Value estimate for this node.

        """
        if self.nn_value_estimate:
            pass
        else:
            self.nn_value_estimate = np.random.uniform(-1, 1)
        return self.nn_value_estimate

    def get_nn_policy_estimate(self) -> dict[str, float]:
        """
        Get policy estimated for this node from neural network.

        Returns
        -------
        dict[str, float]
            Dict of estimated policy, of form {move uci: policy value}.

        """
        if self.nn_policy_estimate:
            pass
        else:
            policy = np.random.rand(len(list(self.board.legal_moves)))
            policy /= policy.sum()
            self.nn_policy_estimate = {
                move.uci(): pol for move, pol in zip(self.board.legal_moves, policy)
            }
        return self.nn_policy_estimate

    def calculate_puct(self) -> None:
        """
        Calculate PUCT in order to estimate best move.

        """
        if self.parent is None:
            pass

        elif self.visits == 0:
            self.puct = (
                self.C
                * self.parent_policy_estimate
                * np.sqrt(np.log(self.parent.visits))
            )

        else:
            first_term = self.value / self.visits
            second_term = (
                self.C
                * self.parent_policy_estimate
                * np.sqrt(np.log(self.parent.visits) / (1 + self.visits))
            )
            self.puct = first_term + second_term

    def find_best_child(self) -> tuple[str, Node]:
        """
        Return child with best puct value.

        Returns
        -------
        tuple[str, Node]
            Returns best move (uci) and best child node.

        """
        if not self.is_leaf:
            pucts = self.get_child_pucts()
            best_ind = np.argmax(list(pucts.values()))
            best_action = list(pucts.keys())[best_ind]
            return best_action, self.children[best_action]
        else:
            raise AttributeError("Node is leaf, has no children. Expand first.")

    def get_child_pucts(self) -> dict[str, float]:
        """
        Get PUCTs of children.

        Returns
        -------
        dict[str, float]
            Dict of form {move uci: puct}.

        """
        if not self.is_leaf:
            return {action: child.puct for action, child in self.children.items()}
        else:
            raise AttributeError("Node is leaf, has no children. Expand first.")

    def get_child_values(self) -> dict[str, float]:
        """
        Get values of children.

        Returns
        -------
        dict[str, float]
            Dict of form {move uci: value}.

        """
        if not self.is_leaf:
            return {action: child.value for action, child in self.children.items()}
        else:
            raise AttributeError("Node is leaf, has no children. Expand first.")

    def get_child_visits(self) -> dict[str, float]:
        """
        Get number of visits of children.

        Returns
        -------
        dict[str, float]
            Dict of form {move uci: visits}.

        """
        if not self.is_leaf:
            return {action: child.visits for action, child in self.children.items()}
        else:
            raise AttributeError("Node is leaf, has no children. Expand first.")


class Game:
    """
    MCTS game.

    """

    def __init__(self, root_state: Optional[str] = None) -> None:
        """
        Initialize mcts game.

        Parameters
        ----------
        root_state : str, optional
            FEN of initital state. The default is None, which corresponds to a
            starting board.

        """
        self.root_state = root_state
        self.reset()

        self.TEMPERATURE = 0  # filled with load_hyperparameter
        self.PLAYS = 0  # filled with load_hyperparameter
        load_hyperparameter(self, "MCTS")

    def reset(self) -> None:
        """
        Reset game: set move counter to 0, initialize root node.

        """
        self.move_counter = 0
        if self.root_state is None:
            self.root = Node(Board().fen())
        else:
            self.root = Node(self.root_state)

    def play(self, **kwargs: Any) -> tuple[list, list, NDArray]:
        """
        Play game until end and save board states, policies, target values.

        Parameters
        ----------
        **kwargs : Any
            Additional parameter passed to choose_move.

        Returns
        -------
        tuple[list, list, list]
            Lists of game relevant information: board states, policies, target values.

        """
        game_states = []
        policies = []

        node = self.root  # create initial node
        if not node.is_leaf:
            raise RuntimeError("Reset game first.")

        game_states.append(node.board.fen())
        while True:
            action, policy = self.choose_move(node, **kwargs)
            node = self.make_move(node, action)
            self.move_counter += 1

            policies.append(policy)
            game_states.append(node.board.fen())

            if node.game_over:
                policies.append({})  # to have same length as game_states
                break

        target_values = self.get_target_values(node)
        return game_states, policies, target_values

    def choose_move(
        self,
        node: Node,
        mode: str = "stochastic",
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """
        Choose move from policies. Mode can be "stochastic" or "deterministic".

        Parameters
        ----------
        node : Node
            The currently active node.
        mode : str, optional
            If "stochastic", choose move randomly from possible moves weighted by
            policy values. If "deterministic", choose move with highest policy value.
            The default is "stochastic".
        **kwargs : any
            Additional parameter passed to get_policy.

        Returns
        -------
        tuple(str, dict)
            Returns choosen move, and policy dict of form {move uci: policy value}.

        """
        policy = self.get_policy(node, **kwargs)

        if mode == "stochastic":
            action = np.random.choice(list(policy.keys()), p=list(policy.values()))
        elif mode == "deterministic":
            action = list(policy.keys())[np.argmax(list(policy.values()))]
        else:
            raise ValueError("mode not known.")
        return action, policy

    def make_move(self, node: Node, action: str) -> Node:
        """
        Make move by choosing child node corresponding to chosen action.

        Parameters
        ----------
        node : Node
            The parent node.
        action : str
            The chosen move action.

        Returns
        -------
        Node
            Child node corresponding to the move action.

        """
        return node.children[action]

    def get_policy(
        self,
        node: Node,
    ) -> dict[str, float]:
        """
        Get policy by performing random playouts based on the number of visits of the
        child nodes

        Parameters
        ----------
        node : Node
            The current node.

        Returns
        -------
        dict[str, float]
            Dict of policy, of form {move uci: policy value}.

        """
        # perform playouts
        for _ in range(self.PLAYS):
            self.selection(node)

        child_visits = node.get_child_visits()
        # calculate policies based on child node visits.
        policy = np.power(list(child_visits.values()), 1 / self.TEMPERATURE)
        policy /= np.sum(policy)  # normalise
        policy_dict = {
            move_uci: policy_value
            for move_uci, policy_value in zip(child_visits.keys(), policy)
        }
        return policy_dict

    def get_target_values(self, node: Node) -> NDArray:
        """
        Calculate target values for every board state: 1 for wins, -1 for losses, 0 for
        draws.

        Parameters
        ----------
        node : Node
            The final node of the game.

        Returns
        -------
        NDArray
            Array with target values.

        """
        last_turn = node.board.turn
        winner = node.outcome.winner

        target_values = np.zeros(self.move_counter + 1).astype(int)
        if winner is None:
            pass
        else:
            if (self.move_counter + 1) % 2:
                target_values[::2] = 1
                target_values[1::2] = -1
            else:
                target_values[::2] = -1
                target_values[1::2] = 1

            # not really needed since game always ends on turn of loser
            if last_turn != winner:
                target_values *= -1
        return target_values

    def selection(self, node: Node) -> float:
        """
        Recursively select nodes, based on PUCT. If leaf node is encountered, expand
        node. If node is end node (game over), return value based on outcome.

        Parameters
        ----------
        node : Node
            The current node.

        Returns
        -------
        float
            The value associated with the node.

        """
        node.visits += 1

        if node.game_over:
            if node.outcome.winner is None:  # draw
                value = 0.0
            elif node.board.turn == node.outcome.winner:  # current player wins
                value = 1.0
            else:  # opponent wins
                value = -1.0

        else:
            if node.is_leaf:
                value = node.get_nn_value_estimate()
                node.expand()
            else:
                best_action, best_child = node.find_best_child()
                best_child.fill(best_action)
                value = self.selection(best_child)

        node.value += value
        if node != self.root:
            node.calculate_puct()
        return -value

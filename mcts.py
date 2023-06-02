#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:24:02 2023

@author: chris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:22:40 2023

@author: chris
"""
import numpy as np
from chess import Board, Move

from typing import Any


class Node:
    def __init__(
        self, state: str, parent=None, policy_probability=1, training_node=False
    ) -> None:
        self.state = state

        if self.state is not None:
            self.board = Board(self.state)
            self.game_over = self.board.is_game_over()

        self.parent = parent
        self.children = {}
        self.legal_actions = ()

        self.is_leaf = True

        self.winner = None

        self.visits = 0
        self.total_reward = 0
        self.policy_probability = policy_probability

        self.puct = None
        if not training_node:
            self.calculate_puct()

    def fill(self, action):
        self.make_board()
        self.make_move(action)
        self.set_state()
        self.check_game_over()

    def make_board(self):
        if self.parent.board.fen() == self.parent.state:
            self.board = self.parent.board
        else:
            self.board = Board(self.parent.state)

    def make_move(self, action):
        self.board.push(Move.from_uci(action))

    def set_state(self):
        self.state = self.board.fen()

    def check_game_over(self):
        self.game_over = self.board.is_game_over()

    def expand(self, **kwargs):
        policy = self.get_nn_policy()
        for i, move in enumerate(self.board.legal_moves):
            self.children[move.uci()] = Node(
                None, parent=self, policy_probability=policy[i], **kwargs
            )
        self.legal_actions = list(self.children.keys())
        self.is_leaf = False

    def get_value(self):
        return np.random.uniform(-1, 1)

    def get_nn_policy(self):
        pol = np.random.rand(len(list(self.board.legal_moves)))
        return pol / pol.sum()

    def calculate_puct(self, c=1):
        if self.parent is None:
            pass
        elif self.visits == 0:
            self.puct = (
                c * self.policy_probability * np.sqrt(np.log(self.parent.visits))
            )
        else:
            first_term = self.total_reward / self.visits
            second_term = (
                c
                * self.policy_probability
                * np.sqrt(np.log(self.parent.visits) / (1 + self.visits))
            )
            self.puct = first_term + second_term

    def find_best_child(self):
        if not self.is_leaf:
            pucts = self.get_child_pucts()
            return list(self.children.items())[np.argmax(pucts)]

    def get_child_pucts(self):
        if not self.is_leaf:
            return np.array([child.puct for child in self.children.values()])

    def get_child_rewards(self):
        if not self.is_leaf:
            return np.array([child.total_reward for child in self.children.values()])

    def get_child_visits(self):
        if not self.is_leaf:
            return np.array([child.visits for child in self.children.values()])


class Game:
    def __init__(self, root_state=None) -> None:
        self.root_state = root_state
        self.reset()

    def play(self, **kwargs: Any) -> tuple[list, list, list]:
        game_states = []
        policies = []

        node = self.root

        game_states.append(node.board.fen())
        while True:
            action, policy = self.choose_move(node, **kwargs)
            node = self.make_move(node, action)
            self.move_counter += 1

            policies.append(policy)
            game_states.append(node.board.fen())

            if node.board.is_game_over():
                policies.append({})  # append empty distribution
                break

        target_values = self.get_target_values(node)
        self.reset()
        return game_states, policies, target_values

    def reset(self):
        self.move_counter = 0
        if self.root_state is None:
            self.root = Node(Board().fen())
        else:
            self.root = Node(self.root_state)

    def choose_move(self, node, mode="stochastic", **kwargs):
        policy = self.get_policy(node, **kwargs)

        if mode == "stochastic":
            action = np.random.choice(list(policy.keys()), p=list(policy.values()))
        elif mode == "deterministic":
            action = list(policy.keys())[np.argmax(list(policy.values()))]
        else:
            return ValueError("mode not known.")
        return action, policy

    def make_move(self, node, action):
        return node.children[action]

    def get_policy(self, node, temperature=1, plays=25):
        for i in range(plays):
            self.selection(node)
        policy = np.power(node.get_child_visits(), 1 / temperature)
        policy /= np.sum(policy)  # normalise
        policy = {
            move_uci: policy_value
            for move_uci, policy_value in zip(node.legal_actions, policy)
        }
        return policy

    def get_target_values(self, node):
        last_turn = node.board.turn
        winner = node.board.outcome().winner

        if winner is None:
            target_values = np.zeros(self.move_counter + 1).astype(int)
        else:
            target_values = np.empty((self.move_counter + 1,)).astype(int)
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

    def selection(self, node=None):
        if node is None:
            node = self.root

        node.visits += 1

        if node.game_over:
            if node.board.outcome().winner is None:  # draw
                value = 0
            elif node.board.turn == node.board.outcome().winner:  # current player wins
                value = 1
            else:  # opponent wins
                value = -1

        else:
            if node.is_leaf:
                value = node.get_value()
                node.expand()
            else:
                best_action, best_child = node.find_best_child()
                best_child.fill(best_action)
                value = self.selection(best_child)

        node.total_reward += value
        if node != self.root:
            node.calculate_puct()
        return -value
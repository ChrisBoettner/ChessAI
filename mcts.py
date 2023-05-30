#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:22:40 2023

@author: chris
"""
import chess
from chess import Board, Move
from scipy.special import softmax   
import numpy as np
import random

class Node:
    def __init__(self, state:str, parent=None, policy_probability=1) -> None:
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
        self.calculate_puct()
    
    def fill(self, action):
        self.make_board()
        self.make_move(action)
        self.set_state()
        self.check_game_over()
        
    def make_board(self):
        if self.parent.board.epd() == self.parent.state:
            self.board = self.parent.board
        else:
            self.board = Board(self.parent.state)
        
    def make_move(self, action):
        self.board.push(Move.from_uci(action))
    
    def set_state(self):
        self.state = self.board.epd()
        
    def check_game_over(self):
        self.game_over = self.board.is_game_over()
    
    def expand(self):
        policy = self.get_policy()
        for i, move in enumerate(self.board.legal_moves):
            self.children[move.uci()] = Node(None, parent=self,
                                             policy_probability=policy[i])            
        self.is_leaf = False
    
    def get_value(self):
        return np.random.uniform(-1,1)
    
    def get_policy(self):
        pol = np.random.rand(len(list(self.board.legal_moves)))
        return pol/pol.sum()
    
    def calculate_puct(self, c=1):
        if self.parent is None:
            pass
        elif self.visits == 0:
            self.puct = (c 
                         * self.policy_probability 
                         * np.sqrt(np.log(self.parent.visits)))
        else:
            first_term  = self.total_reward/self.visits
            second_term = (c * self.policy_probability
                           * np.sqrt(np.log(self.parent.visits)/(1+self.visits)))
            self.puct = first_term + second_term
            
            
    def find_best_child(self):
        if not self.is_leaf:
            pucts  = self.get_child_pucts()
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
    def __init__(self, engine='stockfish') -> None: 
        self.root = Node(Board().epd())
        
        if engine == 'stockfish':
            self.engine = (chess.engine.SimpleEngine
                            .popen_uci(r"stockfish/stockfish_15.1_x64_bmi2"))
        
    def play(self, **kwargs):
        game_states = []
        policies    = []
        
        node = self.root
        i = 0
        
        game_states.append(node.state)
        while True:
            if i%100 == 0:
                print(i) 
            i += 1
            _, node, policy = self.choose_move(node)
            policies.append(policy)
            game_states.append(node.state)
            if node.game_over:
                break
        target_values = self.get_target_values(node, len(game_states))
        
        return game_states, policies, target_values 

    def choose_move(self, node, mode='stochastic', **kwargs):
        policy = self.get_policy_stockfish(node, **kwargs)
        breakpoint()
        if mode=='stochastic':
            action, new_node = random.choices(list(node.children.items()), 
                                            policy, k=1)[0]

        elif mode=='deterministic':
            action, new_node = list(node.children.items())[np.argmax(policy)]            
        return action, new_node, policy
            
    @staticmethod
    def get_policy(node, temperature=1, plays=25):
        for i in range(plays):
            game.selection(node)
        policy = node.get_child_visits()**temperature
        return policy
    
    def get_policy_stockfish(self, node, time_limit=.01):
        scores = []
        for move in node.board.legal_moves:
            info = self.engine.analyse(node.board, 
                                        chess.engine.Limit(time=time_limit),
                                        root_moves=[move])
            scores.append(info["score"].relative.score())
        return softmax(scores)
    
    @staticmethod
    def get_target_values(node, number_of_moves):
        last_turn = node.board.turn
        winner    = node.board.outcome().winner
        
        if winner is None:
            target_values = np.zeros(number_of_moves)
        else:
            target_values = np.empty((number_of_moves,)).astype(int)
            if number_of_moves%2:
                target_values[::2] = 1
                target_values[1::2] = -1
            else:
                target_values[::2] = -1
                target_values[1::2] = 1
                
            if last_turn!=winner:
                target_values *= -1
        
        return target_values
                
    def selection(self, node=None):
        if node is None:
            node = self.root
            
        node.visits +=1
            
        if node.game_over:
            if node.board.outcome().winner is None: # draw
                value =  0
            elif node.board.turn==node.board.outcome().winner: # current player wins
                value = 1
            else: # opponent wins
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
        if node!=self.root:
            node.calculate_puct()
        return -value
         
if __name__=='__main__':
    game = Game()
    a, b, c = game.play()
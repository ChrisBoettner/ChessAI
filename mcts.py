#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:22:40 2023

@author: chris
"""
from chess import Board
import numpy as np
        

class Node:
    def __init__(self, state:str, parent=None, policy_probability=1) -> None:
        self.state = state
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
    
    def expand(self):
        policy = self.get_policy()
        for i, move in enumerate(self.board.legal_moves):
            new_state = self.get_next_state(move)
            self.children[move.uci()] = Node(new_state, parent=self,
                                             policy_probability=policy[i])
            
        self.is_leaf = False
            
    def get_next_state(self, move) -> str:
        self.board.push(move) # make move
        new_state = self.board.fen() # get fen
        self.board.pop() # undo move
        return new_state
    
    def get_value(self):
        return np.random.uniform(-1,1)
    
    def get_policy(self):
        pol = np.random.rand(len(list(self.board.legal_moves)))
        return pol/pol.sum()
    
    def calculate_puct(self, c=1):
        if self.parent is None:
            pass
        elif self.visits == 0:
            self.puct = c * self.policy_probability * np.sqrt(np.log(self.parent.visits))
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
            return [child.puct for child in self.children.values()]
        
    def get_child_rewards(self):
        if not self.is_leaf:
            return [child.total_reward for child in self.children.values()]
        
    def get_child_visits(self):
        if not self.is_leaf:
            return [child.visits for child in self.children.values()]
    
    
class GameTree:
    def __init__(self) -> None: 
        self.root = Node(Board().fen())

    def selection(self, node=None):
        if node is None:
            node = self.root
            
        if node.game_over:
            if node.board.outcome().winner is None: # draw
                value =  0
            elif node.board.turn==node.board.outcome().winner: # current player wins
                value = 1
            else: # opponent wins
                value = -1
        
        else:
            node.visits +=1
            
            if node.is_leaf:
                value = node.get_value()
                node.expand()
            else:
                best_action, best_child = node.find_best_child() 
                value = self.selection(best_child)
                
        node.total_reward += value
        if node!=self.root:
            node.calculate_puct()
        return -value
         
game = GameTree()
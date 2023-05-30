#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:22:40 2023

@author: chris
"""
from chess import Board, Move
import chess.engine
from scipy.special import softmax   
import numpy as np

class Node:
    def __init__(self, state:str, parent=None, policy_probability=1, 
                 training_node=False) -> None:
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
    
    def expand(self, **kwargs):
        policy = self.get_nn_policy()
        for i, move in enumerate(self.board.legal_moves):
            self.children[move.uci()] = Node(None, parent=self,
                                             policy_probability=policy[i],
                                             **kwargs)
        self.legal_actions = list(self.children.keys())    
        self.is_leaf = False
    
    def get_value(self):
        return np.random.uniform(-1,1)
    
    def get_nn_policy(self):
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
    def __init__(self, root_state = None) -> None: 
        self.root_state = root_state
        self.reset()
        
        if root_state is None:
            self.root = Node(Board().epd())
        else:
            self.root = Node(root_state)
        
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
            _, node, policy = self.choose_move(node, **kwargs)
            policies.append(policy)
            game_states.append(node.state)
            if node.game_over:
                break
        target_values = self.get_target_values(node, len(game_states))
        
        return game_states, policies, target_values 
    
    def reset(self):
        if self.root_state is None:
            self.root = Node(Board().epd())
        else:
            self.root = Node(self.root_state)

    def choose_move(self, node, mode='stochastic', **kwargs):
        policy = self.get_policy(node, **kwargs)

        if mode=='stochastic':
            action, new_node = np.random.choices(list(node.children.items()), 
                                            p = policy)
        elif mode=='deterministic':
            action, new_node = list(node.children.items())[np.argmax(policy)]  
        else:
            return ValueError("mode not known.")    
        
        return action, new_node, policy
            
    @staticmethod
    def get_policy(node, temperature=1, plays=25):
        for i in range(plays):
            game.selection(node)
        policy = np.power(node.get_child_visits(), 1/temperature)
        return policy
    
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
    
    
class TrainingGame:
    def __init__(self, root_state=None, engine='stockfish', engine_skill=18) -> None: 
        self.root_state = root_state
        self.reset()
        
        if engine == 'stockfish':
            self.engine = (chess.engine.SimpleEngine
                           .popen_uci(r"stockfish/stockfish_15.1_x64_bmi2"))
            
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
                    engine_skill = (self.random_skill_level(),
                                    self.random_skill_level())
                case _ :
                    raise ValueError("engine_skill input not understood.")
                    
            self.engine_skill = engine_skill
        
    @staticmethod
    def random_skill_level(power=-0.2):
        levels = np.arange(1,21)
        weights = np.abs((levels-np.mean(levels)))**power
        return(np.random.choice(levels, p = weights/sum(weights)))   
     
    def play(self, root_state=None, **kwargs):
        game_states = []
        policies    = []
        
        node = self.root
        
        game_states.append(node.state)
        while True:
            _, node, policy = self.choose_move(node, **kwargs)
            policies.append(policy)
            game_states.append(node.state)
            if node.game_over:
                break
        target_values = self.get_target_values(node, len(game_states))
        return game_states, policies, target_values 
    
    def reset(self):
        self.move_counter = 0
        
        if self.root_state is None:
            self.root = Node(Board().epd(), training_node=True)
        else:
            self.root = Node(self.root_state, training_node=True)

    def choose_move(self, node, mode='stochastic', **kwargs):
        node.expand(training_node=True)
        policy = self.get_policy_stockfish(node, **kwargs)
        
        if mode=='stochastic':
            action = np.random.choice(list(policy.keys()),p=list(policy.values()))
        elif mode=='deterministic':
            action = list(policy.keys())[np.argmax(list(policy.values()))]
        else:
            return ValueError("mode not known.")
            
        new_node = node.children[action]           
        new_node.fill(action)
        self.move_counter +=1
        
        return action, new_node, policy
    
    def get_policy_stockfish(self, node, time_limit=.01, temperature=1, 
                             mate_score=int(1e+5)):
        infos = self.engine.analyse(node.board, chess.engine.Limit(time=time_limit), 
                                    multipv=500, 
                                    options = {"Skill Level": 
                                               self.engine_skill[self.move_counter%2]})

        scores = [info["score"].relative.score(mate_score=mate_score) for info in infos]
        scores = softmax(scores)**(1/temperature)
        scores = scores/np.sum(scores)
        
        scores = {info["pv"][0].uci():score for info, score in zip(infos,scores)}
        return scores
    
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
         
if __name__=='__main__':
    game = TrainingGame()
    k = 0
    for i in range(30):
        a, b, c = game.play()
        game.reset()
        if sum(c)==0:
            k +=1
            
    # Skill level might not work, since main difference in skill levels seems to be that
    # it takes non-ideal moves as evaluated by analyse function, but we don't use the
    # plays picked by the engine, so that will have no effect
    
    # probably better to increase temperature in order to increase randomness 
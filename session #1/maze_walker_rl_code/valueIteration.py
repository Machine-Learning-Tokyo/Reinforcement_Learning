import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output,display
import time

from environment import *

class ValueIteration:
    def __init__(self,discount=0.9,maximum_iteration=100000,display_mode="graphic",exec_mode="user_input"):
        self.env = Env()
        self.discount = discount
        self.maximum = maximum_iteration
        self.sleep_time = 0.5
        self.states = [(x,y) for y in range(0,BOARD_SIZE[0]) for x in range(0,BOARD_SIZE[0])]
        self.display_mode=display_mode
        self.exec_mode = exec_mode
        #self.display(self.env.state_value,self.env.policy)
        
    def display(self,state_value,policy):
        if self.display_mode == "graphic":
            self.env.draw_board(state_value,policy)
        elif self.display_mode == "text":
            print("Current State values:")
            self.env.draw_state_value(state_value)
            print("\nCurrent policy:")
            self.env.draw_policy(policy)
            print("\n----------------------------------------------")
        else:
            raise Exception(str("Unknown display mode"))
    
    def run(self):
        policy_prob = 1.0
        env_prob = 1.0
        itr = 0
        self.display(self.env.state_value,self.env.policy)
        while itr < self.maximum:
            itr+=1
            save_state_value = self.env.state_value.copy()
            for s in self.states:                                             # we iterate over all the states to evaluate their state_values
                if s not in OBSTABLES.values() and s not in GOAL.values():
                    action_choose_dict = {}
                    for a in ACTION_SPACE.keys():                             # evaluate each possible action in that state (right/left/up/down)
                        next_state, reward = self.env.get_next_state_and_reward(s,a)
                        # Measuring/saving the current + future rewards (discounted) for the current state + action pair.
                        action_choose_dict[a] = np.round(policy_prob * env_prob * (reward + self.discount*self.env.state_value[next_state]),2)
                    
                    # Use a "GREEDY" approach to select the best action for this state (which maximizes the current + future rewards)
                    
                    self.env.policy[s] = sorted(action_choose_dict.keys(), key=lambda x : action_choose_dict[x])[-1]
                    next_state, reward = self.env.get_next_state_and_reward(s,self.env.policy[s])
                    
                    # Based on best action selected, update the current state value based on best action possible.
                    save_state_value[s] = np.round(policy_prob * env_prob * (reward + self.discount*self.env.state_value[next_state]),2)
            if np.sum(save_state_value == self.env.state_value) == BOARD_SIZE[0] * BOARD_SIZE[1]:
                
                # Check if further improvement happened in this iteration, if not terminate and mark done.
                print("ENDING NOW : Total Number of Iterations : %d" %(itr))
                del(save_state_value)
                self.display(self.env.state_value,self.env.policy)
                break
            else:
                # If update of state_values happened, copy the state_values and continue the learning process.
                del(self.env.state_value)
                self.env.state_value = save_state_value.copy()
                del(save_state_value)
                self.display(self.env.state_value,self.env.policy)
            if self.exec_mode == "user_input":
                inp = input("At iteration %d" %(itr))
            else:
                time.sleep(self.sleep_time)
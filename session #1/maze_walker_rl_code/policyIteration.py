import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output,display
import time

from environment import *

class PolicyIteration:
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
        run_policy_improvement = True
        while run_policy_improvement:
            # Policy improvement step
            itr = 0
            while itr < self.maximum:
                # Policy Evaluation step
                itr += 1

                save_state_value = self.env.state_value.copy()
                for s in self.states:
                    if s not in OBSTABLES.values() and s not in GOAL.values():
                        s_v = self.env.state_value[s]
                        action = self.env.policy[s]
                        next_state, reward = self.env.get_next_state_and_reward(s,action)

                        #print("current state = %s, current_state_value : %d, current_action = %s, next_state = %s, reward = %d" %(str(s),s_v,action,str(next_state),reward))

                        save_state_value[s] = np.round(policy_prob * env_prob * (reward + self.discount*self.env.state_value[next_state]),2)
                        #inp= input("ABCD")
                if np.sum(self.env.state_value == save_state_value) == BOARD_SIZE[0] * BOARD_SIZE[1]:
                    #print(itr)
                    del(save_state_value)
                    break
                else:
                    del(self.env.state_value)
                    self.env.state_value = save_state_value.copy()
                    del(save_state_value)
            
            save_policy = self.env.policy.copy()
            for s in self.states:
                if s not in OBSTABLES.values() and s not in GOAL.values():
                    action_choose_dict = {}
                    for a in ACTION_SPACE.keys():
                        next_state, reward = self.env.get_next_state_and_reward(s,a)
                        action_choose_dict[a] = np.round(policy_prob * env_prob * (reward + self.discount*self.env.state_value[next_state]),2)
                    save_policy[s] = sorted(action_choose_dict.keys(), key=lambda x : action_choose_dict[x])[-1]
            
            print("Policy Evaluation worked for : %d iterations. Now the state value and policy are:" %(itr))
            self.display(self.env.state_value,self.env.policy)
            if np.sum(save_policy == self.env.policy) == BOARD_SIZE[0] * BOARD_SIZE[1]:
                run_policy_improvement=False
                del(save_policy)
            else:
                del(self.env.policy)
                self.env.policy = save_policy.copy()
                del(save_policy)
            
            if self.exec_mode == "user_input":
                inp = input("At iteration %d" %(itr))
            else:
                time.sleep(self.sleep_time)

        print("--end--")
        self.display(self.env.state_value,self.env.policy)
        
    def run_policy_evaluation_example(self,iteration=1):
        self.display(self.env.state_value,self.env.policy)
        policy_prob = 1.0
        env_prob = 1.0
        itr = 0
        while itr < iteration:
            # Policy Evaluation step
            itr += 1

            save_state_value = self.env.state_value.copy()
            for s in self.states:
                if s not in OBSTABLES.values() and s not in GOAL.values():
                    s_v = self.env.state_value[s]
                    action = self.env.policy[s]
                    next_state, reward = self.env.get_next_state_and_reward(s,action)

                    #print("current state = %s, current_state_value : %d, current_action = %s, next_state = %s, reward = %d" %(str(s),s_v,action,str(next_state),reward))

                    save_state_value[s] = np.round(policy_prob * env_prob * (reward + self.discount*self.env.state_value[next_state]),2)
                    #inp= input("ABCD")
            if np.sum(self.env.state_value == save_state_value) == BOARD_SIZE[0] * BOARD_SIZE[1]:
                #print(itr)
                del(save_state_value)
                break
            else:
                del(self.env.state_value)
                self.env.state_value = save_state_value.copy()
                del(save_state_value)
        self.display(self.env.state_value,self.env.policy)
        print("Total Iterations executed : %d" %(itr))
        
    def run_policy_improvement_example(self):
        self.display(self.env.state_value,self.env.policy)
        save_policy = self.env.policy.copy()
        policy_prob = 1.0
        env_prob = 1.0
        for s in self.states:
            if s not in OBSTABLES.values() and s not in GOAL.values():
                action_choose_dict = {}
                for a in ACTION_SPACE.keys():
                    next_state, reward = self.env.get_next_state_and_reward(s,a)
                    action_choose_dict[a] = np.round(policy_prob * env_prob * (reward + self.discount*self.env.state_value[next_state]),2)
                save_policy[s] = sorted(action_choose_dict.keys(), key=lambda x : action_choose_dict[x])[-1]
        if np.sum(save_policy == self.env.policy) == BOARD_SIZE[0] * BOARD_SIZE[1]:
            run_policy_improvement=False
            del(save_policy)
        else:
            del(self.env.policy)
            self.env.policy = save_policy.copy()
            del(save_policy)
        self.display(self.env.state_value,self.env.policy)
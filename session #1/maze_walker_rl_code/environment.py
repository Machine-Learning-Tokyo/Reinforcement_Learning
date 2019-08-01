import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output,display
import time
np.seterr(invalid="ignore")
BOARD_SIZE = (4,4)

ACTION_SPACE = {"UP" : (-1,0),# Defined as (reduction/addition in rows , reduction/addition in columns)
                "DOWN" : (1,0),
                "RIGHT" : (0,1),
                "LEFT" : (0,-1)
               }

OBSTABLES = {"MAN" :(2,1),
             "TREE":(1,3)
            }

GOAL = {"GOAL":(3,3)}

class Env:
    def __init__(self):
        self.policy, self.state_value = self._initBoard()   
        
    def get_next_state_and_reward(self,state,action):
        action_movement = ACTION_SPACE[action]
        n_state = (state[0] + action_movement[0],state[1] + action_movement[1])
        
        if n_state[0] < 0 or n_state[1] < 0 or n_state[0] > BOARD_SIZE[0]-1 or n_state[1] > BOARD_SIZE[0]-1:
            # If by taking this action, we are pushed outside the board, then we remain the current state, but get a reward = -1
            n_state = state
            return n_state, -1
        
        elif n_state in OBSTABLES.values():
            # if we get to an obstable by taking this action, then we reamin current state, but reward = -10
            n_state = state
            return n_state,-10
        
        elif n_state == GOAL.values():
            # if we get to the goal by taking this action, then reward = +10
            return n_state,10
        
        else:
            # In all other cases, we get a reward of -1
            return n_state,-1
    
    def _initBoard(self):
        _s_v = np.zeros(BOARD_SIZE)
        _p_action = np.array(random.choices(list(ACTION_SPACE.keys()),
                                     k=BOARD_SIZE[0]*BOARD_SIZE[1])).reshape(BOARD_SIZE)
        for k, v in OBSTABLES.items():
            _p_action[v] = k
        
        for k, v in GOAL.items():
            _p_action[v] = k     
        return _p_action, _s_v
    
    def reset(self):
        self.policy, self.state_value = self._initBoard()
        
    def draw_policy(self,policy):
        print(policy)
    
    def draw_state_value(self,state_value):
        print(state_value)
        
    def draw_board(self,state_value,policy):
        #clear_output()
        all_states = [(x,y) for y in range(0,BOARD_SIZE[0]) for x in range(0,BOARD_SIZE[0])]
        x = np.zeros((state_value.shape[0],state_value.shape[1],3))
        x[:,:,0] = state_value
        x = (x / x.min()) #* 255
        
        for x_cord,y_cord in all_states:
            if (x_cord,y_cord) in GOAL.values():
                x[x_cord,y_cord,1] = 1.0
            elif (x_cord,y_cord) in OBSTABLES.values():
                x[x_cord,y_cord,2] = 1.0
            else:
                pass
        
        fig, ax = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(10,5))
        ax[0].imshow(x,interpolation="nearest")
        for x_cord,y_cord in all_states:
            ax[0].text(x=x_cord,y=y_cord,s=str(state_value.T[x_cord,y_cord]),horizontalalignment="center",verticalalignment="center",fontsize=15,color="white")
    
        ax[1].imshow(x,interpolation="nearest")
        for x_cord,y_cord in all_states:
            ax[1].text(x=x_cord,y=y_cord,s=policy.T[x_cord,y_cord],horizontalalignment="center",verticalalignment="center",fontsize=15,color="white")
        
        ax[0].set_title("State Values")
        ax[1].set_title("Policy")
        
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].axes.get_yaxis().set_visible(False)
        
        ax[1].axes.get_xaxis().set_visible(False)
        ax[1].axes.get_yaxis().set_visible(False)
        
        
        
        display(plt.show())
        return
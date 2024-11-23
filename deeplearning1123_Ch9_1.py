import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6 # 狀態個數，每個狀態都有 ACTIONS 的機率
ACTIONS = ['left', 'right'] #動作
EPSILON = 0.9   # The probability of choosing the optimal strategy, 剩0.1是用猜的 (試著跳出 q talbe 思維)
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # decay ratio
MAX_EPISODES = 13   # Maximum number of rounds
FRESH_TIME = 0.3    # Travel interval time

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     
        columns=actions,    # actions's name
    )
    return table

tmp_table = build_q_table(N_STATES,ACTIONS)
print(tmp_table)

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]#Select all the action values of this state    
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()): #10% chance, or the action value of the state is 0
        action_name = np.random.choice(ACTIONS) #Randomly pick an action
    else:
        action_name = state_actions.idxmax()       # Select the action with the maximum value
    return action_name

def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate 
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # - - - - -T is the environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='') #/r : Return the cursor to the beginning
        time.sleep(2)
        print('\n                                ', end='') #end='':No line breaks after output
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list) # Make the list to be a string        
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)            
            S_, R = get_env_feedback(S, A)  # Achieve the next status and score
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal': 
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  
            S = S_  

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

"""
Created on Mon Jun 24 16:29:35 2019

Create a basic test bed for the Chase gym environment...
"""
import numpy as np
import copy

from datetime import datetime

def write_chase_log(log, agent_name):
    file_time = datetime.now().strftime("%Y%m%d - %H%M")
    f_name = 'Chase - ' + agent_name + ' - ' + file_time + '.csv'
    with open(f_name, "w", newline="") as f:
        h_line = 'Episode,Step,Action,Reward,Done,{}\n'.format(np.array2string(np.arange(0.0, 400.0)).replace('.', ',').replace('\n', '').replace(' ', '')[1:-2])
        f.write(h_line)
        for i in range(len(log)):
            l_line = '{},{},{},{},{},{}\n'.format(log[i][0],
                                                  log[i][1],
                                                  log[i][2],
                                                  log[i][3],
                                                  log[i][4],
                                                  str(log[i][5]).replace('.', ',').replace('\n', '')[1:-2]) 
            f.write(l_line)
        f.close
       

import gym
#from gym_chase.envs import ChaseEnv

#env = gym.make('gym_chase:Chase-v0')

# Simple human agent

EPISODES = 100
e = 0
state_log = []

# Simple human agent

while e < EPISODES:
    done = False
    e_step = 0
    total_reward = 0
    state = env.reset(random_seed=e)
    state = state.ravel()
    
    state_log.append([e, e_step, None, None, done, copy.deepcopy(state)])
    
    while not done:
        env.render()
        print('\n7   8   9')
        print('  \\ | /')
        print('4 - 5 - 6')
        print('  / | \\')
        print('1   2   3')
        p_move = input('\nYour move [1-9 move, 5 stay still]:')
        n_state, r, done = env.step(p_move)
        print('\nEpisode:', e, 'Step:', e_step)
        print('\nReward:', r)
        total_reward += r
        e_step += 1
        n_state = n_state.ravel()
        state_log.append([e, e_step, p_move, r, done, copy.deepcopy(n_state)])
    env.render()
    if total_reward == 5:
        print("\nAll robots eliminated. Total reward =", total_reward)
    else:
        print("\nAgent eliminated. Total reward =", total_reward)        
    e += 1
    
write_chase_log(state_log, 'Human')

# Simple random agent
"""
import random

while e < EPISODES:
    done = False
    e_step = 0
    total_reward = 0
    state = env.reset(random_seed=e)
    state = state.ravel()
    state_log.append([e, e_step, None, None, done, copy.deepcopy(state)])
    
    random.seed()
    while not done:
        # time.sleep(2)
        env.render()
        move = random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        n_state, r, done = env.step(move)
        print('\nReward:', r)
        total_reward += r 
        e_step += 1
        n_state = n_state.ravel()
        state_log.append([e, e_step, move, r, done, copy.deepcopy(n_state)])
    env.render()
    if total_reward == 5:
        print("\nAll robots eliminated. Total reward =", total_reward)
    else:
        print("\nAgent eliminated. Total reward =", total_reward)
    e += 1
    
write_chase_log(state_log, 'Random')
"""
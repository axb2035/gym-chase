"""
Created on Mon Jun 24 16:29:35 2019

Create a basic test bed for the Chase gym environment...
"""
import random
import time

import gym
from gym_chase.envs import ChaseEnv

env = gym.make('gym_chase:Chase-v0')

# Simple human agent

EPISODES = 3
e = 0

# Simple human agent

while e < EPISODES:
    done = False
    e_step = 1
    total_reward = 0
    env.reset(random_seed=e)    
    while not done:
        env.render()
        print('\n7   8   9')
        print('  \\ | /')
        print('4 - 5 - 6')
        print('  / | \\')
        print('1   2   3')
        p_move = input('\nYour move [1-9 move, 5 stay still]:')
        _, r, done = env.step(p_move)
        print('\nEpisode:', e, 'Step:', e_step)
        print('\nReward:', r)
        total_reward += r
        e_step += 1
    env.render()
    if total_reward == 5:
        print("\nAll robots eliminated. Total reward =", total_reward)
    else:
        print("\nAgent eliminated. Total reward =", total_reward)
    e += 1

# Simple random agent
"""
random.seed()
done = False
while not done:
    # time.sleep(2)
    env.render()
    move = random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    _, r, done = env.step(move)
    print('\nReward:', r)
    total_reward += r 

env.render()

if total_reward == 5:
    print("\nAll robots eliminated. Total reward =", total_reward)
else:
    print("\nAgent eliminated. Total reward =", total_reward)
"""


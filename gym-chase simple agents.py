"""
Created on Mon Jun 24 16:29:35 2019

Create a basic test bed for the Chase gym environment...
"""
import random
import time

import gym
from gym_chase.envs import ChaseEnv

env = gym.make('gym_chase:Chase-v0')

done = False
total_reward = 0


# Simple human agent

while not done:
    env.render()
    move = input('\nYour move [1-9 move, 5 stay still]:')
    _, r, done = env.step(move)
    print('\nReward:', r)
    total_reward += r 


# Simple random agent
"""
random.seed()
while not done:
    # time.sleep(2)
    env.render()
    move = random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    _, r, done = env.step(move)
    print('\nReward:', r)
    total_reward += r 
"""

env.render()
if total_reward == 5:
    print("\nAll robots eliminated. Total reward =", total_reward)
else:
    print("\nAgent eliminated. Total reward =", total_reward)
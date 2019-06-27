import sys

from gym import utils
from gym.envs.toy_text import discrete

import numpy as np
import random

SW = 1
S = 2
SE = 3
W = 4
NM = 5
E = 6
NW = 7
N = 8
NE = 9

def generate_arena(robots=5, random_seed=0):
    random.seed(random_seed)
    ax, ay = 20, 20
    arena = np.zeros((ax, ay))
        
    # Create boundary zappers.
    for i in range(ax):
        for j in range(ay):            
            if j in [0, ay-1] or i in [0, ax-1]:
               arena[i][j] = 1

    # Place random zappers.
    z = 0
    while z < 10:
        z_x, z_y = random.randint(1,19), random.randint(1,19)
        if arena[z_x][z_y] == 0:
            arena[z_x][z_y] = 2
            z += 1

    # Place killer robots.
    r_pos = []
    r = 0
    while r < robots:
        r_x, r_y = random.randint(1,19), random.randint(1,19)
        if arena[r_x][r_y] == 0:
            arena[r_x][r_y] = 3
            r_pos.append([r_x, r_y])
            r += 1

    # Place agent.
    a = 0
    while a < 1:
        a_x, a_y = random.randint(1,19), random.randint(1,19)
        if arena[a_x][a_y] == 0:
            arena[a_x][a_y] = 4
            a += 1
            a_pos = [a_x, a_y]

    return arena, r_pos, a_pos

class ChaseEnv(discrete.DiscreteEnv):
    """
    Chase is based on a text game first created in the 1970's and featured
    in a number of 80s personal computer programming books.
    
    The world is a 20x20 arena surrounded by high voltage zappers. Ten random
    zappers are also distributed around the arena. If the agent moves into
    a zapper (either by moving to an outside edge of arena or into a free 
    standing one) it is eliminated and the epsisode ends.
    
    Besides the zappers there are also five robots which move towards the
    agent each step. The robots have no self-preservation instincts and will
    move into a zapper in an attempt to get closer to the agent. If a robot 
    moves into the same square as the agent the agent is eliminated and the
    episode ends. If a robot wants to move to a square which is occupied
    by another robot it will not move.
    
    Each step an agent can move horiziontally one square, vertically one 
    square, a combination of one vertcal and horiziontal square or not move.
    This gives the agent nine possible actions per step.
    
    An example map looks like this:
        
    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 3. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 3. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 2. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 2. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 4. 0. 0. 3. 1.
    1. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 1.
    1. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
    
    1 : Boundary zapper
    2 : Random zapper
    3 : Robot
    4 : Agent
    
    The aim of the game is for the agent to eliminate the robots by placing 
    a zapper between the agent and robot so the robot moves into a zapper in 
    an attempt to capture the agent.

    The episode ends when:
        - the agent is eliminated by moving into a zapper;
        - the agent is eliminated by a robot moving into the agent; or
        - all robots are eliminated.
        
    The agent receives a reward of 1 for each robot eliminated and zero 
    otherwise.

    """
    
    metadata = {'render.modes': ['human']}
    
    #def __init__(self):
   
def test_init:
    shape = (20, 20)
    arena, r_pos, p_pos = generate_arena()
    
    nS = (shape[0]-2) * (shape[1]-2) * 4
    nA = 9
    
    arena_vec = arena.ravel()

        
        
        def step(self, action):
        ...
        def reset(self):
        ...
def test_render(self, mode='human'):
    arena, r_pos, p_pos = generate_arena()
    outfile = sys.stdout
   
    arena_human = np.array2string(arena)
    arena_human = np.char.replace(arena_human, '0.', ' ')
    arena_human = np.char.replace(arena_human, '1.', 'X')
    arena_human = np.char.replace(arena_human, '2.', 'X')
    arena_human = np.char.replace(arena_human, '3.', 'R')
    arena_human = np.char.replace(arena_human, '4.', 'A')
    
    output = ''.join(arena_human.ravel())
    
    outfile.write(output)
    

    

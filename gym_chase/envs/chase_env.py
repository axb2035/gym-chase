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
    
    def __init__(self):
        self.shape = (20, 20)
        self.arena, self.r_pos, self.a_pos = generate_arena()
        
        self.nS = (self.shape[0]-2) * (self.shape[1]-2) * 4
        self.nA = 9
        
        # Thinking I should take this out - it's up to the agent to work
        # out what it wants to do with the state. I shouldn't assume the 
        # agent wants it back as a 1D vector.
        self.arena_vec = self.arena.ravel()

        
        
    def step(self, action):
        r = 0
        a_pos = self.a_pos
        r_pos = self.r_pos
        done = False
        
        if action == '7':
            # Move North West.
            a_move = [1, -1]    
        elif action == '8':
            # Move North.
            a_move = [1, 0]
        elif action == '9':
            # Move North East.
            a_move = [1, 1]
        elif action == '6':
            # Move East.
            a_move = [0, 1]
        elif action == '3':
            # Move South East.
            a_move = [-1, 1]
        elif action == '2':
            # Move South.
            a_move = [-1, 0]
        elif action == '1':
            # Move South West.
            a_move = [-1, -1]
        elif action == '4':
            # Move West.
            a_move = [0, -1]            
        elif action == '5':
            # If I stay still maybe they won't see me!
            a_move = [0, 0]
        else: # Should not get here - treat as no move.
            a_move = [0, 0]
            
        # assess the move
        if self.arena[a_pos[0] + a_move[0]][a_pos[1] + a_move[1]] in [1, 2, 3]:
            # ZZZAAAAPPPPP!!!! - agent ran into a boundary, zapper or robot.
            done = True
        
        # Move agent (vacate location and set new location).
        self.arena[a_pos[0]][a_pos[1]] = 0
        self.arena[a_pos[0] + a_move[0]][a_pos[1] + a_move[1]] = 4
        a_pos[0] += a_move[0]
        a_pos[1] += a_move[1]
        
        # Robots turn!
        robot = 0
        while robot < len(r_pos):
            # Which way to the player?
            tar_x, tar_y = a_pos[0] - r_pos[robot][0], a_pos[1] - r_pos[robot][1]            
            
            if abs(tar_x) == abs(tar_y): 
                r_move = [np.sign(tar_x), np.sign(tar_y)]
            elif abs(tar_x) > abs(tar_y):
                r_move = [np.sign(tar_x), 0]
            else:
                r_move = [0, np.sign(tar_y)]

            # Check to make sure robots don't move on top of each other.
            if self.arena[r_pos[robot][0] + r_move[0]][r_pos[robot][1] + r_move[1]] == 3:
                r_move = [0, 0]
            
            # Has robot caught the player?
            if self.arena[r_pos[robot][0] + r_move[0]][r_pos[robot][1] + r_move[1]] == 4:
                # ZZZAAAAPPPPP!!!! - Agent was caught by a robot.
                done = True
            
            # Check if robot done something stupid otherwise update position.
            if self.arena[r_pos[robot][0] + r_move[0]][r_pos[robot][1] + r_move[1]] in [1, 2]:
                if len(r_pos)-1 == 0:
                    self.arena[r_pos[robot][0]][r_pos[robot][1]] = 0
                    # ZZZAAAAPPPPP!!!! - All robots gone. Agent wins.
                    r += 1
                    done = True
                else:
                    # ZZZAAAAPPPPP!!!! - Fried robot.
                    self.arena[r_pos[robot][0]][r_pos[robot][1]] = 0
                    del r_pos[robot]   
                    r += 1
            else: # Update robot position
                self.arena[r_pos[robot][0]][r_pos[robot][1]] = 0
                self.arena[r_pos[robot][0] + r_move[0]][r_pos[robot][1] + r_move[1]] = 3
                r_pos[robot][0] += r_move[0]
                r_pos[robot][1] += r_move[1]
            
            robot += 1
                
        return self.arena, r, done

    def reset(self):
        self.arena, self.r_pos, self.a_pos = generate_arena()
        return


    def render(self, mode='human'):
        outfile = sys.stdout
       
        arena_human = np.array2string(self.arena)
        arena_human = np.char.replace(arena_human, '0.', ' ')
        arena_human = np.char.replace(arena_human, '1.', 'X')
        arena_human = np.char.replace(arena_human, '2.', 'X')
        arena_human = np.char.replace(arena_human, '3.', 'R')
        arena_human = np.char.replace(arena_human, '4.', 'A')
        
        output = ''.join(arena_human.ravel())
        
        outfile.write(output)
    

# Testing...

env = ChaseEnv()    

done = False

env.render()

while not done:
    move = input('\nYour move [1-9 move, 5 stay still]:')
    _, r, done = env.step(move)
    env.render()
    print('\nReward:', r)
    

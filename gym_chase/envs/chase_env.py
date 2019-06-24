from gym import utils
from gym.envs.toy_text import discrete

import numpy as np

SW = 1
S = 2
SE = 3
W = 4
NM = 5
E = 6
NW = 7
N = 8
NE = 9


class ChaseEnv(discrete.DiscreteEnv):
    """
    Chase is based on a text game first created in the 1970's and featured
    in a number of 80s personal computer programming books.
    
    The world is a 20x20 grid surrounded by high voltage zappers. Ten random
    zappers are also distributed around the world. If the agent moves into
    a zapper (either by moving off the world or into a free standing one) it
    is eliminated and the epsisode ends.
    
    Besides the zappers there are also five robots which move towards the
    agent each step. The robots have no self-preservation instincts and will
    move into a zapper in an attempt to get closer to the agent. If a robot 
    moves into the same square as the agent the agent is eliminated and the
    episode ends.
    
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
    
    The aim of the game is for the agent to get the robots to destruct by
    moving into a zapper in an attempt to capture the agent.

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
    
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...


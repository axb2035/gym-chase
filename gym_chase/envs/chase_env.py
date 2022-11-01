import sys
import numpy as np
import random

import gym
from gym import spaces


def generate_arena(robots=5, random_seed=0):
    random.seed(random_seed)
    ax, ay = 20, 20
    arena = np.zeros((ax, ay), dtype=np.uint8)

    # Create boundary zappers.
    for i in range(ax):
        for j in range(ay):
            if j in [0, ay-1] or i in [0, ax-1]:
                arena[i][j] = 1

    # Place random zappers.
    z = 0
    while z < 10:
        z_x, z_y = random.randint(1, 19), random.randint(1, 19)
        if arena[z_x][z_y] == 0:
            arena[z_x][z_y] = 2
            z += 1

    # Place killer robots.
    r_pos = []
    r = 0
    while r < robots:
        r_x, r_y = random.randint(1, 19), random.randint(1, 19)
        if arena[r_x][r_y] == 0:
            arena[r_x][r_y] = 3
            r_pos.append([r_x, r_y])
            r += 1

    # Place agent.
    a = 0
    while a < 1:
        a_x, a_y = random.randint(1, 19), random.randint(1, 19)
        a += 1
        if arena[a_x][a_y] == 0:
            # Check for neigbouring robots.
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if arena[a_x + x][a_y + y] == 3:
                        a = 0
        else:
            a = 0
    arena[a_x][a_y] = 4
    a_pos = [a_x, a_y]

    return arena, r_pos, a_pos


def look(arena, loc, tar):
    return arena[loc[0] + tar[0]][loc[1] + tar[1]]


def move(arena, loc, tar, element):
    arena[loc[0]][loc[1]] = 0
    arena[loc[0] + tar[0]][loc[1] + tar[1]] = element


class ChaseEnv(gym.Env):
    """
    Chase is based on a text game first created in the 1970's and featured
    in a number of 1980's personal computer programming books. See:
    https://www.atariarchives.org/morebasicgames/showpage.php?page=26
    for an example.

    The arena is a 20x20 arena surrounded by high voltage zappers. Ten random
    zappers are also distributed around the arena. If the agent moves into
    a zapper (either by moving to an outside edge of arena or into a free
    standing one) it is eliminated and the epsisode ends.

    Each step an agent can move horiziontally one square, vertically one
    square, a combination of one vertcal and horiziontal square or not move.
    This gives the agent nine possible actions per step.

    Besides the zappers there are also five robots which move towards the
    agent each step. The robots have no self-preservation instincts and will
    move into a zapper in an attempt to get closer to the agent. If a robot
    moves into the same square as the agent the agent is eliminated and the
    episode ends. If a robot wants to move to a square which is occupied
    by another robot it will not move. If the agent moves into a zapper the
    robots will still move completing the 'step'.

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

    The agent receives a reward of 1 for each robot eliminated, -1 if the agent
    is eliminated and zero otherwise.

    """

    metadata = {'render_modes': ['human'], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0,
                                            high=4,
                                            shape=(20, 20),
                                            dtype=np.uint8)

        self.action_space = spaces.Discrete(9,)

        self.action_to_direction = {
            1: np.array([-1, 1]),
            2: np.array([0, 1]),
            3: np.array([1, 1]),
            4: np.array([-1, 0]),
            5: np.array([0, 0]),
            6: np.array([1, 0]),
            7: np.array([-1, -1]),
            8: np.array([0, -1]),
            9: np.array([1, -1]),
        }

    def step(self, action):
        r = 0
        a_pos = self.a_pos
        r_pos = self.r_pos
        done = False
        a_move = self.action_to_direction[action]

        # Assess agent move.
        if look(self.arena, a_pos, a_move) in [1, 2, 3]:
            # ZZZAAAAPPPPP!!!! - agent ran into a boundary, zapper or robot.
            self.arena[a_pos[0]][a_pos[1]] = 0
            r -= 1
            done = True
        else:
            # Move agent (vacate location and set new location).
            move(self.arena, a_pos, a_move, element=4)

        # Even if zapped, need to update agent for possible pyhrric reward.
        a_pos[0] += a_move[0]
        a_pos[1] += a_move[1]

        # Robots turn!
        robot = 0
        r_del = []
        while robot < len(r_pos):
            # Which way to the player?
            tar_x = a_pos[0] - r_pos[robot][0]
            tar_y = a_pos[1] - r_pos[robot][1]

            if abs(tar_x) == abs(tar_y):
                r_move = [np.sign(tar_x), np.sign(tar_y)]
            elif abs(tar_x) > abs(tar_y):
                r_move = [np.sign(tar_x), 0]
            else:
                r_move = [0, np.sign(tar_y)]

            tar_look = look(self.arena, r_pos[robot], r_move)

            # Check to make sure robots don't move on top of each other.
            if tar_look == 3:
                r_move = [0, 0]

            # Has robot caught the player?
            if tar_look == 4:
                # ZZZAAAAPPPPP!!!! - Agent was caught by a robot.
                r -= 1
                done = True

            # Check if robot done something stupid otherwise update position.
            if tar_look in [1, 2]:
                # ZZZAAAAPPPPP!!!! - Fried robot.
                self.arena[r_pos[robot][0]][r_pos[robot][1]] = 0
                r_del.append(robot)
                r += 1
            else:  # Update robot position.
                move(self.arena, r_pos[robot], r_move, element=3)
                r_pos[robot][0] += r_move[0]
                r_pos[robot][1] += r_move[1]
            robot += 1

        # Clean out eliminated robots.
        r_adj = 0
        for r_dead in r_del:
            del r_pos[r_dead - r_adj]
            r_adj += 1
        if len(r_pos) == 0:
            done = True  # All robots eliminated.

        # TODO: Add info. For now return an empty dict.
        info = {}

        return self.arena, r, done, False, info

    def reset(self, seed=None, options=None):
        self.arena, self.r_pos, self.a_pos = generate_arena(random_seed=seed)
        # TODO: Add info. For now return an empty dict.
        info = {}

        return self.arena, info

    def render(self):
        outfile = sys.stdout

        arena_human = np.array2string(self.arena)

        arena_human = np.char.replace(arena_human, '0', ' ')
        arena_human = np.char.replace(arena_human, '1', 'X')
        arena_human = np.char.replace(arena_human, '2', 'X')
        arena_human = np.char.replace(arena_human, '3', 'R')
        arena_human = np.char.replace(arena_human, '4', 'A')
        arena_human = np.char.replace(arena_human, '[', '')
        arena_human = np.char.replace(arena_human, ']', '')

        output = ' ' + '\n'.join(arena_human.ravel())

        outfile.write(output)

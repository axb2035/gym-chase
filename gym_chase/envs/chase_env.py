import sys
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces


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

    X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X
    X  X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  R  .  .  X
    X  .  A  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  X  .  .  .  .  .  .  .  X
    X  .  .  X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X  .  X
    X  .  .  .  .  .  .  .  .  R  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  X  .  .  .  .  .  .  X
    X  .  .  .  X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  R  .  .  .  .  .  X
    X  .  .  .  .  R  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  X  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  X  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X  .  .  .  X
    X  .  .  .  .  .  .  X  .  .  .  .  .  .  .  .  .  .  .  X
    X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  R  X
    X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X

    X : Boundary zapper
    X : Random zapper
    R : Robot
    A : Agent
    . : Empty

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
        self.size = 20

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size-1, shape=(2,), dtype=int),
                "robots": spaces.Box(0, self.size-1, shape=(2, 5), dtype=int),
                "zappers": spaces.Box(0, self.size-1, shape=(2, 10), dtype=int)
            }
        )

        self.action_space = spaces.Discrete(9,)

        self.action_to_direction = {
            1: np.array([1, -1]),
            2: np.array([1, 0]),
            3: np.array([1, 1]),
            4: np.array([0, -1]),
            5: np.array([0, 0]),
            6: np.array([0, 1]),
            7: np.array([-1, -1]),
            8: np.array([-1, 0]),
            9: np.array([-1, 1]),
        }

    def get_keys_to_action(self):
        return self.action_to_direction

    def _generate_arena(self, robots=5, random_seed=0):
        random.seed(random_seed)

        # Place 10 zappers.
        zapper_list = np.empty((0, 2), int)
        while len(zapper_list) < 10:
            loc = np.array([[random.randint(1, 18), random.randint(1, 18)]])
            if loc not in zapper_list:
                zapper_list = np.append(zapper_list, loc, axis=0)

        # Place robots.
        robot_list = np.empty((0, 2), int)
        while len(robot_list) < robots:
            loc = np.array([[random.randint(1, 18), random.randint(1, 18)]])
            if (loc not in zapper_list) and (loc not in robot_list):
                robot_list = np.append(robot_list, loc, axis=0)

        # Place Agent.
        agent_pos = None
        while agent_pos is None:
            location = np.array([random.randint(1, 18), random.randint(1, 18)])
            if (location not in zapper_list) and (location not in robot_list):
                agent_pos = location

        return {"zappers": zapper_list,
                "robots": robot_list,
                "agent": agent_pos}

    def _look(self, arena, loc, tar):
        return arena[loc[0] + tar[0]][loc[1] + tar[1]]

    def _move(self, arena, loc, tar, element):
        arena[loc[0]][loc[1]] = 0
        arena[loc[0] + tar[0]][loc[1] + tar[1]] = element

    def step(self, action):
        r = 0
        terminated = False
        # Episodes are never truncated.
        truncated = False

        # Move agent.
        self.game_state['agent'] += self.action_to_direction[action]

        # Assess agent move - did it run into a boundary, zapper or robot?
        if self.game_state['agent'][0] in [0, 19]:
            terminated = True
        if self.game_state['agent'][1] in [0, 19]:
            terminated = True
        if (self.game_state['agent'] == self.game_state['zappers']).all(1).any():
            terminated = True
        if (self.game_state['agent'] == self.game_state['robots']).all(1).any():
            terminated = True

        if terminated:
            # ZZZAAAAPPPPP!!!! - agent ran into the boundary, zapper or robot.
            r = -1

        # Robots turn!
        # Even if Agent dies, complete step for possible pyhrric reward.
        robot = 0
        robot_del = list()
        a_pos = self.game_state['agent']

        # Iterate through robots moving and assessing.
        while robot < len(self.game_state['robots']):
            r_pos = self.game_state['robots'][robot]

            # Which way to the agent?
            tar_x = a_pos[0] - r_pos[0]
            tar_y = a_pos[1] - r_pos[1]

            if abs(tar_x) == abs(tar_y):
                r_move = np.array([np.sign(tar_x), np.sign(tar_y)])
            elif abs(tar_x) > abs(tar_y):
                r_move = np.array([np.sign(tar_x), 0])
            else:
                r_move = np.array([0, np.sign(tar_y)])

            # Commit move if not moving onto another robot.
            r_clash = r_pos + r_move

            if not (r_clash == self.game_state['robots']).all(1).any():
                r_pos += r_move
                self.game_state['robots'][robot] = r_pos

            # Has robot caught the player?
            if np.array_equal(r_pos, a_pos):
                # ZZZAAAAPPPPP!!!! - Agent caught by Robot.
                # No extra reward penalty if Agent already zapped.
                if not terminated:
                    r -= 1
                    terminated = True

            # Check if robot has done something stupid.
            r_zapped = False
            if r_pos[0] in [0, 19]:
                r_zapped = True
            elif r_pos[1] in [0, 19]:
                r_zapped = True
            elif (r_pos == self.game_state['zappers']).all(1).any():
                r_zapped = True

            if r_zapped:
                # ZZZAAAAPPPPP!!!! - Fried robot.
                robot_del.append(True)
                r += 1
            else:
                robot_del.append(False)

            robot += 1

        # Clean out eliminated robots.
        self.game_state['robots'] = np.delete(self.game_state['robots'], robot_del, 0)

        # All robots eliminated? Game over!
        if len(self.game_state['robots']) == 0:
            terminated = True

        observation = self.game_state
        # TODO: Add info. For now return an empty dict.
        info = {}

        return observation, r, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.game_state = self._generate_arena(random_seed=seed)

        observation = self.game_state
        # TODO: Add info. For now return an empty dict.
        info = {}

        return observation, info

    def render(self):
        outfile = sys.stdout
        arena = list()

        # Add the boundaries and fill others positions with spaces.
        for x in range(self.size):
            row = list()
            for y in range(self.size):
                if x in [0, 19]:
                    row.append("X")
                elif y in [0, 19]:
                    row.append("X")
                else:
                    row.append(".")
            arena.append(row)

        # Plug in the positions of the agent, robots and zappers.
        a = self.game_state['agent']
        arena[a[0]][a[1]] = "A"
        for r in self.game_state['robots']:
            arena[r[0]][r[1]] = "R"
        for z in self.game_state['zappers']:
            arena[z[0]][z[1]] = "X"
        output = '\n'.join(['  '.join([col for col in row]) for row in arena])

        outfile.write(output)

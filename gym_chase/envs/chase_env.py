"""Gymnasium gym-chase toy_text environment."""
import copy
import random
import sys
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete


class ChaseEnv(gym.Env):
    """gym-chase is a toy_text environment for the gymnasium RL library.

    Chase is based on a text game first created in the 1970's and featured
    in a number of 1980's personal computer programming books. See:
    https://www.atariarchives.org/morebasicgames/showpage.php?page=26
    for an example.

    The arena is a 20x20 arena surrounded by high voltage zappers. Ten random
    zappers are also distributed around the arena. If the agent moves into
    a zapper (either by moving to an outside edge of arena or into a free
    standing one) it is eliminated and the episode ends.

    Each step an agent can move horizontally one square, vertically one
    square, a combination of one vertical and horizontal square or not move.
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

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        """Setup action and observation spaces, default keymap and arena size."""
        self.render_mode = render_mode
        self.size = 20
        self.robots = 5
        self.zappers = 10

        robot_space = Dict()
        for r in range(self.robots):
            robot_space[r] = Dict(
                {
                    "alive": Discrete(2),
                    "location": Box(0, self.size - 1, shape=(2,), dtype=np.int32),
                }
            )

        zapper_space = Dict()
        for z in range(self.zappers):
            zapper_space[z] = Box(0, self.size - 1, shape=(2,), dtype=np.int32)

        self.observation_space = Dict(
            {
                "agent": Box(0, self.size - 1, shape=(2,), dtype=np.int32),
                "robots": robot_space,
                "zappers": zapper_space,
            }
        )

        self.action_space = Discrete(9)

        self.action_to_direction = {
            0: np.array([1, -1]),
            1: np.array([1, 0]),
            2: np.array([1, 1]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
            5: np.array([0, 1]),
            6: np.array([-1, -1]),
            7: np.array([-1, 0]),
            8: np.array([-1, 1]),
        }

    def get_keys_to_action(self) -> Dict:
        """Return default keymap for chase."""
        return {key + 1: value for key, value in self.action_to_direction.items()}

    def _generate_arena(self, random_seed: Optional[int] = 0) -> np.ndarray:
        """Generates a random valid map.

        Args:
            random_seed: seed used to generate arena. Allows consistent sequence generation.

        Returns:
            A random valid map
        """
        random.seed(random_seed)

        # Place 10 zappers.
        zapper_list = np.empty((0, 2), int)
        while len(zapper_list) < 10:
            loc = np.array([[random.randint(1, 18), random.randint(1, 18)]])
            if loc not in zapper_list:
                zapper_list = np.append(zapper_list, loc, axis=0)

        # Place robots.
        robot_list = np.empty((0, 2), int)
        while len(robot_list) < self.robots:
            loc = np.array([[random.randint(1, 18), random.randint(1, 18)]])
            if (loc not in zapper_list) and (loc not in robot_list):
                robot_list = np.append(robot_list, loc, axis=0)

        # Place agent.
        agent_pos = None
        while agent_pos is None:
            location = np.array([random.randint(1, 18), random.randint(1, 18)])
            if (location not in zapper_list) and (location not in robot_list):
                agent_pos = location

        # Convert lists to expected observation space.
        robot_list = dict(enumerate([{"alive": 1, "location": r} for r in robot_list]))
        zapper_list = dict(enumerate(zapper_list))

        return {"agent": agent_pos, "robots": robot_list, "zappers": zapper_list}

    def step(
        self, action: int, project: Optional[bool] = False
    ) -> Tuple[Dict, int, bool, bool, Dict]:
        """Move agent based on action, move robots in response and assess outcomes."""
        projected_state = copy.deepcopy(self.game_state)
        r = 0
        terminated = False
        # Episodes are never truncated. Unless there is a wrapper with a timer.
        truncated = False

        # Move agent.
        projected_state["agent"] += self.action_to_direction[action]
        a_pos = projected_state["agent"]

        # Assess agent move - did it run into a boundary, zapper or robot?
        if a_pos[0] in [0, 19]:
            # Ran into boundary
            terminated = True
        elif a_pos[1] in [0, 19]:
            # Ran into boundary
            terminated = True
        elif any([(a_pos == z).all() for z in self.game_state["zappers"].values()]):
            # "Ran into zapper"
            terminated = True
        elif any(
            [(a_pos == r["location"]).all() for r in self.game_state["robots"].values()]
        ):
            terminated = True

        # Even if Agent dies, complete step for possible pyrrhic reward.

        # # Iterate through robots moving and assessing.
        for robot in projected_state["robots"].values():
            if robot["alive"]:
                r_pos = robot["location"]

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

                if not any(
                    [
                        (r_clash == r["location"]).all() and r["alive"]
                        for r in projected_state["robots"].values()
                    ]
                ):
                    r_pos += r_move
                    robot["location"] = r_pos

                # Has robot caught the player?
                if np.array_equal(r_pos, a_pos):
                    # ZZZAAAAPPPPP!!!! - Agent caught by Robot.
                    terminated = True

                # Check if robot has done something stupid.
                r_zapped = False
                if r_pos[0] in [0, 19]:
                    r_zapped = True
                elif r_pos[1] in [0, 19]:
                    r_zapped = True
                elif any(
                    [(r_pos == z).all() for z in projected_state["zappers"].values()]
                ):
                    r_zapped = True

                if r_zapped:
                    # ZZZAAAAPPPPP!!!! - Fried robot.
                    robot["alive"] = 0
                    r += 1

        # If the episode has been terminated then we know if the agent has
        # been eliminated by moving into a zapper or robot, or the agent
        # been caught by a robot.
        if terminated:
            r -= 1

        # TODO: All robots eliminated? Game over!
        if sum(r["alive"] for r in self.game_state["robots"].values()) == 0:
            terminated = True

        if not project:
            # If the step is not a projection then update the game state.
            # print("Updating real state.")
            self.game_state = projected_state
            observation = self.game_state
        else:
            # If the step is a projection then return the projected state.
            # print("Projecting state.")
            observation = projected_state

        # print("Observation: ", observation["agent"], r)
        # Indicate if the result is a projective state. i.e. it wasn't stepped forward.
        info = {"project": project}

        return observation, r, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Returns a new arena based on random_seed."""
        self.game_state = self._generate_arena(random_seed=seed)

        observation = self.game_state
        # TODO: Add info. For now return an empty dict.
        info = {}

        return observation, info

    def render(self, render_state: Optional[Dict] = None) -> None:
        """Outputs a text representation of the observation space."""
        if render_state is None:
            render_state = self.game_state
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
        a = render_state["agent"]
        arena[a[0]][a[1]] = "A"
        for r in render_state["robots"].values():
            arena[r["location"][0]][r["location"][1]] = "R"
        for z in render_state["zappers"].values():
            arena[z[0]][z[1]] = "X"
        output = "\n".join(["  ".join([col for col in row]) for row in arena])

        outfile.write(output)

    def get_state(self) -> Dict:
        """Returns the current game state."""
        return self.game_state

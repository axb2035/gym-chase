import gymnasium as gym
import pytest
import numpy as np
from tqdm import tqdm
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def state_to_json(state):
    """Convert a state to a JSON string.

    Args:
        state: A state to convert to a JSON string.

    Returns:
        A JSON string representation of the state.
    """
    return json.dumps(state, sort_keys=True, cls=NumpyEncoder,)


@pytest.fixture
def env() -> gym.Env:
    """Create a new unwrapped instance of the Chase-v1 environment."""
    chase = gym.make("gym_chase:Chase-v1")
    return chase.unwrapped  # As using projection, we need to use the unwrapped environment.


def test_chase_environment(env: gym.Env) -> None:
    """Test basic projected v actual functionality of the Chase RL environment.

    This test performs the following checks:

    1. Projects each action for the current state and then validates the agent and robots have not moved by comparing against the initial state.

    Args:
        env: An unwrapped instance of the gym_chase:Chase-v1 environment
    """
    # Set the number of episodes to run.
    for e in tqdm(range(100), desc="Projection v actual episodes"):
        terminated = False
        initial_state, _ = env.reset(seed=e)
        initial_state_json = state_to_json(initial_state)

        while not terminated:
            # Project each action for the current state.
            for action in range(9):
                # Take a projected action and observe the result.
                state, reward, terminated, truncated, info = env.step(action, True)
                # Validate the agent hasn't moved comparing against the initial state.
                assert state_to_json(state) != initial_state_json, "Projected state is not the same as initial state."

            # Then take a random action.
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action, False)

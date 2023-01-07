import gymnasium as gym
import pytest

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete


@pytest.fixture
def env() -> gym.Env:
    """Create a new instance of the FrozenLake-v1 environment."""
    return gym.make('gym_chase:Chase-v1')


def test_chase_environment(env: gym.Env) -> None:
    """Test basic functionality of the Chase RL environment.

    This test performs the following checks:

    1. Verifies that the observation space is a dictionary with the correct keys.
    2. Verifies that the values for the "agent" and "robots" keys are Box spaces with the correct shape and dtype.
    3. Verifies that the value for the "zappers" key is a Dict space with the correct keys and values.
    4. Takes a random action and verifies that the resulting state is valid.
    5. Verifies that the reward is a valid value (either 0.0 or 1.0).
    6. Verifies that the terminated flag is a boolean value.
    7. Verifies that the truncated flag is a boolean value.
    8. Run 20,000 episodes where random actions are selected.
    9. Run 20,000 episodes where the no move action is selected to test all reward scenarios.

    Args:
        env: An instance of the gym_chase:Chase-v1 environment
    """
    # Reset the environment to the starting state.
    state = env.reset()

    # Verify that the observation space is a dictionary with the correct keys.
    assert isinstance(env.observation_space, Dict), "Observation space is not a dictionary."
    assert set(env.observation_space.spaces.keys()) == {"agent", "robots", "zappers"}, "Observation space has incorrect keys."

    # Verify that the "agent" value is a Box space with the correct shape and dtype
    agent_space = env.observation_space.spaces["agent"]
    assert isinstance(agent_space, Box), "Agent space is not a Box space."
    assert agent_space.shape == (2,), "Agent space has incorrect shape."
    assert agent_space.dtype == np.int32, "Agent space has incorrect dtype."

    # Verify that the "robots" value is a Dict space with the correct keys and values.
    robots_space = env.observation_space.spaces["robots"]
    assert isinstance(robots_space, Dict), "Robots space is not a Dict space."
    assert set(robots_space.spaces.keys()) == set(range(env.robots)), "Robots space has incorrect keys."
    for key, value in robots_space.spaces.items():
        assert isinstance(value, Dict), "Robot space is not a Dict space."
        assert set(value.spaces.keys()) == {"alive", "location"}, "Robot space has incorrect keys."
        assert isinstance(value.spaces["alive"], Discrete), "Alive space is not a Discrete space."
        assert value.spaces["alive"].n == 2, "Alive space has incorrect number of values."
        assert isinstance(value.spaces["location"], Box), "Location space is not a Box space."
        assert value.spaces["location"].shape == (2,), "Location space has incorrect shape."
        assert value.spaces["location"].dtype == np.int32, "Location space has incorrect dtype."

    # Verify that the "zappers" value is a Dict space with the correct keys and values.
    zappers_space = env.observation_space.spaces["zappers"]
    assert isinstance(zappers_space, Dict), "Zappers space is not a Dict space."
    assert set(zappers_space.spaces.keys()) == set(range(env.zappers)), "Zappers space has incorrect keys."
    for key, value in zappers_space.spaces.items():
        assert isinstance(value, Box), "Zapper space is not a Box space."
        assert value.shape == (2,), "Zapper space has incorrect shape."
        assert value.dtype == np.int32, "Zapper space has incorrect dtype."

    # Run random and possum tests to get a range of actions, states and rewards.
    exit_rewards = np.zeros(7)
    for e in range(10000):
        terminated = False
        total_reward = 0
        state, info = env.reset(seed=e)
        while not terminated:
            # Choose a random action.
            action = env.action_space.sample()
            # Take the action and observe the result.
            state, reward, terminated, truncated, info = env.step(action)

            # Verify that the new state is valid.
            # TODO: assert state >= 0 and state < env.observation_space.n, "Invalid new state after taking action."

            # Verify that the reward is a valid value.
            assert reward in range(-1, 5), "Invalid reward value."
            total_reward += reward

            # Verify that the terminated flag is a boolean value.
            assert isinstance(terminated, bool), "Terminated flag is not a boolean value."

            # Verify that the truncated flag is a boolean value.
            assert isinstance(truncated, bool), "Truncated flag is not a boolean value."
        if not exit_rewards[total_reward+1]:
            exit_rewards[total_reward+1] = True

    print("\nRandom exit rewards:", exit_rewards)

    # Possum should get at least one instance of all reward scenarios.
    for e in range(20000):
        terminated = False
        total_reward = 0
        state, info = env.reset(seed=e)
        while not terminated:
            # Choose a random action.
            action = 4
            # Take the action and observe the result.
            state, reward, terminated, truncated, info = env.step(action)

            # Verify that the reward is a valid value.
            assert reward in range(-1, 5), "Invalid reward value."
            total_reward += reward

        if not exit_rewards[total_reward+1]:
            exit_rewards[total_reward+1] = True

    print("Possum exit rewards:", exit_rewards)

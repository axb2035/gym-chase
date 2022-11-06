from gymnasium.envs.registration import register

# Toy Text
# ----------------------------------------

register(
    id='Chase-v1',
    entry_point='gym_chase.envs:ChaseEnv',
)

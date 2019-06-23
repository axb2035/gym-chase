from gym.envs.registration import register

# Toy Text
# ----------------------------------------

register(
    id='Chase-v0',
    entry_point='gym_chase.envs:ChaseEnv',
)
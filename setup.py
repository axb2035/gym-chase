from setuptools import setup

setup(
    name="gym_chase",
    version="0.0.2",
    packages=["gym_chase", "gym_chase.envs"],
    install_requires=["gym"],  # Add any other dependencies Chase needs
)

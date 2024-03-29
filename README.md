[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![gymnasium](https://img.shields.io/badge/version-0.27-brightgreen.svg?label=gymnasium)](https://gymnasium.farama.org)

# gym-chase
## A toy text gymnasium environment based on Chase.

Chase is based on a text game I first saw in the 1970's on a Nixdorf mini
computer (pretty sure it was a 8870/M55) and featured
in a number of 1980's personal computer programming books. See:
https://www.atariarchives.org/morebasicgames/showpage.php?page=26
for an example.

The challenge is to build a reinforcement learning agent that can consistently
eliminate all robots without getting eliminated itself.

## Installation
To use gym-chase `gymnasium` needs to be installed into your target virtual 
environment. To install gym-chase activate your target virtual environment and
type:
```
> pip install git+https://github.com/axb2035/gym-chase.git
```

## The environment
The environment is a 20x20 arena surrounded by high voltage zappers. Ten 
random zappers are also distributed around the arena. If the agent moves 
into a zapper (either by moving to an outside edge of arena or into a free 
standing one) it is eliminated and the episode ends.

Each step an agent can move horizontally one square, vertically one 
square, a combination of one vertical and horizontal square or not move.
This gives the agent nine possible actions per step.

```
	7  8  9
	 \ | /
	4- 5 -6
	 / | \
	1  2  3
```

Besides the zappers there are also five robots which move towards the
agent each step. The robots have no self-preservation instincts and will
move into a zapper in an attempt to get closer to the agent. 

The agent is eliminated if a robot moves into the same square. If a robot 
tries to move to a square which is occupied by another robot it will not move. 
If the agent moves into a zapper the robots will still move completing the 
'step' for a possible pyrrhic reward.

An example state looks like this:
```
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
```

The aim of the game is for the agent to eliminate the robots by placing 
a zapper between the agent and robot so the robot moves into a zapper in 
an attempt to capture the agent.

The episode ends when:
- the agent is eliminated by moving into a zapper;
- the agent is eliminated by a robot moving into the agent; or
- all robots are eliminated.

The agent receives a reward of 1 for each robot eliminated, -1 if the agent
is eliminated and zero otherwise. The agent elimination penalty is only 
applied once per step.

## Projection feature.
Most `gymnasium` environments advance the state when `step(action)` is called. A non-standard feature of gym-chase is the ability to request a projection of the future state based on an action, that will not advance the underlying state of the environment.

To obtain a projected state call:
```python
env.step(action, project=True)
```

This allows for other types of agents to be tested, such as reflex agents that select the best outcome from available (s, a) pairs to determine which action to take. It will also allow for search algorithms such as MCTS to be used.

By default, Gymnasium environments are wrapped by the `passiveEnvChecker` wrapper, which means that it will throw an error if you try to pass the project argument as the checker enforces a single argument for `step()`. To avoid this problem the unwrapped environment needs to be called when it is made with:
```python
env = gym.make("gym_chase:Chase-v1").unwrapped
```

## Other Notes

When resetting the environment it will generate the same arena every time. If
you want a different setup pass a value, such as the episode number to generate 
a different starting position:
```python
env.reset(seed=101)
```
It may be possible to get a never ending sequence of moves between the agent 
and one remaining robot (though I haven't proven it yet). Recommend putting a 
step ceiling on any agent to ensure episode will end.

## Performance tables
The following are agents that you can benchmark against. The validation set for 
non-human is for the first 10,000 arenas (to set a starting arena see notes 
above). Generated using v1 of the environment.

| Agent   | mean r  | % won (r=5) |
| :-------|:-------:|:-----------:|
| Human<sup>1</sup>  |  4.11  	| 84.0%       |
| Reflex03  | 1.3786 |  20.42%      |
| Reflex02  | 1.4011 |  19.64%      |
| Reflex01  | 1.1244 |  12.13%      |
| Possum  | -0.3342 |  1.06%      |
| Random  | -0.4257 |  0.08%      |

The reflex agents assessed the (s, a) pairs and selected the highest value. Ties were broken randomly.

Reflex01 used the `reward`. This leads to situation where there may be a tie between a move into a safe space and moving into a zapper (-1) and having a robot also follow in (+1).

Reflex02 attempts to mitigate the issues of 01 by using `reward - terminating` value. Moves that cause the agent to die are now less valuable than moves that keep the agent alive. However, ties can still be caused, or in extreme cases the agent throwing itself into a zapper may have the highest value if two or more robots would be taken down with it!

Reflex03 is the last of the sequence and uses `reward - (terminating * 5)` to ensure the agent will always chose a safe move over one that causes it to be eliminated.

1. Based on first 100 arenas. I do have life outside of this project...

## Todo / Future expansion

- Remove human agent from `chase_play` repo and make the env work with 
`play(gymnasium.make('gym_chase:Chase-v1'))`.
- Make everything more 'gymnasiumthonic'.
- Add render option `machine` to omit spaces used for padding on stdout.
- Add `pygame` render option.
- Create different reward function wrappers. 
    - To encourage eliminating the robots as quickly as possible.
    - To make the episode as long as possible ie. can the game go on forever.
- Make the environment more 'stochastic' by adding option for some random 
variation to the robots i.e. 0.1 chance of going to the left or right of the 
'deterministic' move to close the gap on the agent.
- Add option for zappers blocking line of sight (LOS) between agent and robots. 
If robots lose LOS to agent then they will not move or take a random move that doesn't
take them into a zapper.
- Add option for "fog" so the robots position is not exactly known until 
closer to the Agent.
- Add option for the original 'Jump' action where the Agent would be 
teleported to a random square. Generally used when the agent was about to be 
caught. Though this could land the agent on a robot or zapper so it wasn't a 
guaranteed escape plan!
- Add option that robots moving into each other are both eliminated.
- Any improvements in coding that are suggested.

## Thanks to
- Eike and everyone at the Melbourne MLAI Bookclub for insightful feedback.
- The original developers and maintainers of the original gym and new 
gymnasium libraries. :)

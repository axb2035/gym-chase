# gym-chase
## A toy text gym environment based on Chase.

Chase is based on a text game I first saw in the 1970's and featured
in a number of 1980's personal computer programming books. See:
https://www.atariarchives.org/morebasicgames/showpage.php?page=26
for an example.

The challenge is to build a reinforcement learning agent that can consistently
eliminate all robots without getting eliminated itself.

## Installation
To use gym-chase ```gym``` needs to be installed into your target virtual 
environment. To install gym-chase activate your target virtual environment and
type:
```
> pip install git+https://github.com/axb2035/gym-chase.git
```

## The environment
The environment is a 20x20 arena surrounded by high voltage zappers. Ten 
random zappers are also distributed around the arena. If the agent moves 
into a zapper (either by moving to an outside edge of arena or into a free 
standing one) it is eliminated and the epsisode ends.

Each step an agent can move horiziontally one square, vertically one 
square, a combination of one vertcal and horiziontal square or not move.
This gives the agent nine possible actions per step.

	7  8  9
	 \ | /
	4- 5 -6
	 / | \
	1  2  3

Besides the zappers there are also five robots which move towards the
agent each step. The robots have no self-preservation instincts and will
move into a zapper in an attempt to get closer to the agent. If a robot 
moves into the same square as the agent the agent is eliminated and the
episode ends. If a robot wants to move to a square which is occupied
by another robot it will not move. If the agent moves into a zapper the
robots will still move completing the 'step'.    

An example state looks like this:
```
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
```
1. Boundary zapper.
2. Random zapper.
3. Robot.
4. Agent.

The aim of the game is for the agent to eliminate the robots by placing 
a zapper between the agent and robot so the robot moves into a zapper in 
an attempt to capture the agent.

The episode ends when:
- the agent is eliminated by moving into a zapper;
- the agent is eliminated by a robot moving into the agent; or
- all robots are eliminated.

The agent receives a reward of 1 for each robot eliminated, -1 if the agent
is eliminated and zero otherwise.

## Notes

When resetting the environment it will generate the same arena every time. If
you want a different setup pass a value, such as the episode number to generate 
a different starting position:
```
env.reset(random_seed=101)
```
It may be possible to get a never ending sequence of moves between the agent 
and one remaining robot. Recommend putting a step ceiling on any agent to
ensure episode will end.

## Performance tables
The following are agents that you can benchmark against. The validation set for 
non-human is for the first 10,000 arenas (to set a starting arena see notes 
above).

| Agent   | mean r  | % won (r=5) |
| --------|---------|-------------|
| Human*  |  4.11  	| 84.0%       |
| Possum  | -0.3342 |  1.06%      |
| Random  | -0.4257 |  0.08%      |

* Based on first 100 arenas. I have life outside of this project...

## Todo / Future expansion

- Create a different reward function to encourage eliminting the robots as
quickly as possible.
- Make the environment more 'stochastic' by add some random variation to the 
robots i.e. 0.1 chance of going to the left or right of the 'deterministic' 
move to close the gap on the agent.
- Reinstate the original 'Jump' function where you would be transported to a 
random square when the agent had no hope of winning. Though this could land the
agent on a robot or zapper so it wasn't a gaurenteed escape plan!
- Reinstate that robots moving into each other are both eliminated.
- Any improvements in coding that are suggested.

## Thanks to
- Eike and everyone at the Melbourne MLAI Bookclub for insightful feedback.

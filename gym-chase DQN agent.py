# AXB implementation of DQN with Keras
import random
import numpy as np
from collections import deque
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

import gym
from gym_chase.envs import ChaseEnv


class DQNAgent():
    def __init__(self, state_size, action_size):
        
        # Environment parameters.
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience Replay Buffer parameters.
        self.memory = deque(maxlen=2000)
        
        # Discount rate.
        self.gamma = 0.99
        
        # ANN parameters.
        self.learning_rate = 0.001
        self.batch_size = 32
                
        # Exploration v exploitation (Epsilon) parameter.
        self.epsilon = 0.1
        
        # Epsilon decay parameters.
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.999
        
        self.dqn_model = self.build_dqn()

    # Build model.
    # Need to research architecuture - TO DO
    def build_dqn(self):
        dqn_model = Sequential()
        dqn_model.add(Dense(512, input_dim=5, activation='relu'))
        dqn_model.add(Dense(512, activation='relu'))
        dqn_model.add(Dense(512, activation='relu'))
        dqn_model.add(Dense(self.action_size, activation='softmax'))
        dqn_model.compile(loss='mse',
                          optimizer=Adam(lr=self.learning_rate))
        return dqn_model
    
    # Determine an action!
    def action(self, state):
        if random.random() <= self.epsilon:
            act = random.randrange(self.action_size) + 1
            #print('Action (random):', act)
            return act
        else:
            act_values = self.dqn_model.predict(state)
            # Argmax will return value from 0-8 when need 1-9.
            act = np.argmax(act_values[0]) + 1
            #print('Action (model):', act)
            return act
    
    # Manage replay buffer.
    def update_memory(self, state, action, r, n_state, done):
        self.memory.append((state, action, r, n_state, done))
    
    # Update DQN using replay buffer.
    def update_model(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, r, n_state, done in minibatch:
            target = r
            if not done:
                target = (r + self.gamma *
                          np.amax(self.dqn_model.predict(n_state)[0]))
            target_f = self.dqn_model.predict(state)
            target_f[0][action-1] = target
            self.dqn_model.fit(state, target_f, epochs=1, verbose=0)
        # Epsilon decay
        # if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.dqn_model.load_weights(name)

    def save(self, name):
        self.dqn_model.save_weights(name)

# run episodes

env = gym.make('gym_chase:Chase-v0')
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]

agent = DQNAgent(state_size, env.action_space)


EPISODES = 5
e = 1
e_steps = []
t_rewards = []

# Simple DQN agent
start_time = datetime.now()

while e < EPISODES + 1:
    done = False
    e_step = 1
    total_reward = 0
    state = env.reset(e-1).ravel()
    state = to_categorical(state, num_classes=5)

    # Reinitialize random seed.
    random.seed()
#    print('------------------------------')
    #env.render()
    while not done:
        #print('Episode:', e, 'Step:', e_step)
        
        # Choose action based on state.
        action = agent.action(state)
        
        # Send action to environment and get new state.
        n_state, r, done = env.step(str(action))
        n_state = n_state.ravel()
        n_state = to_categorical(n_state, num_classes=5)
        #env.render()
        
        # Update the memory queue.
        agent.update_memory(state, action, r, n_state, done)
        
        # update model/learning
        if len(agent.memory) > agent.batch_size:
                agent.update_model()
        
        # update the old state to the new state
        state = n_state

        # Capture info for future... 
        #print('Reward:', r)
        total_reward += r
        e_step += 1

    if total_reward == 5:
        print("All robots eliminated. Total reward =", total_reward)
    else:
        print("Agent eliminated. Total reward =", total_reward)
    #print('Memory size:', len(agent.memory))
    #print('Epsilon:{:.2}'.format(agent.epsilon))
    print('Episode:', e, 'Steps', e_step)
    e_steps.append(e_step)
    t_rewards.append(total_reward)
    e += 1

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

agent.save('exp_16')

import matplotlib.pyplot as plt

plt.plot(e_steps, label='steps')
plt.plot(t_rewards, label='reward')
plt.yticks(np.arange(min(t_rewards), max(e_steps)+1, step=int((max(e_steps) - min(t_rewards))/10+1)))
plt.legend(loc='upper right')
plt.title('Chase DQN\nepisodes=' + str(EPISODES)
          + ' e=' + str(agent.epsilon) 
          + ' gamma=' + str(agent.gamma) 
          + ' \nlr=' + str(agent.learning_rate) 
          + ' minibatch=' + str(agent.batch_size) + ' replay buffer=' + str(len(agent.memory))
          + '\narch: i512 relu, h512 relu, h512 relu, o9 softmax\nloss=mse, optimiser=adam'
          + '\nDuration: {}'.format(end_time - start_time))

# write stats to file...
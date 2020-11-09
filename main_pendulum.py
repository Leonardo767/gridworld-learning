# ref: https://www.youtube.com/watch?v=cO5g5qLrLSo

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents import DQNAgent
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
import gym
import random


# model
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


env = gym.make('CartPole-v0')
env._max_episode_steps = 500
states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions)
model.summary()
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

train = False
if train:
    dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)
    dqn.save_weights('dqn_weights.h5f', overwrite=True)
else:
    dqn.load_weights('dqn_weights.h5f')
    _ = dqn.test(env, nb_episodes=5, visualize=True)

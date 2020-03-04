#!/usr/bin/env python3

import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory


NUM_EPISODES = 20


if __name__ == "__main__":
    env = gym.make("stocks-v0")
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Option 1 : Simple model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))

    # Option 2: deep network
    #model = Sequential()
    #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    #model.add(Dense(16))
    #model.add(Activation('relu'))
    #model.add(Dense(16))
    #model.add(Activation('relu'))
    #model.add(Dense(16))
    #model.add(Activation('relu'))
    #model.add(Dense(nb_actions))
    #model.add(Activation('softmax'))


    print(model.summary())


    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = EpisodeParameterMemory(limit=1000, window_length=1)

    cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
                   batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
    cem.compile()

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    cem.fit(env, nb_steps=100000, visualize=False, verbose=2)

    # After training is done, we save the best weights.
    cem.save_weights('cem_{}_params.h5f'.format("stocks-v0"), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    cem.test(env, nb_episodes=5, visualize=False)
    
    #for i_episode in range(NUM_EPISODES):
    #    observation = env.reset()
    #    while True:
    #        action = env.action_space.sample()
    #        observation, reward, done, info = env.step(action)
    #        # env.render()
    #        if done:
    #            print("info:", info)
    #            break
    #plt.cla()
    #env.render_all()
    #plt.show()

#!/usr/bin/env python3

from enum import Enum
from trading_env import TradingEnv

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt

import gym
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents import SARSAAgent, DQNAgent
from rl.policy import MaxBoltzmannQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory


Datafiles = ["aapl.us.txt","dis.us.txt","ge.us.txt","ibm.us.txt","intc.us.txt","jpm.us.txt","msft.us.txt","nke.us.txt","v.us.txt","wmt.us.txt"]
ValidatorFiles = ["dis.us.txt","aapl.us.txt","ibm.us.txt","nke.us.txt","v.us.txt","jpm.us.txt","msft.us.txt","wmt.us.txt","intc.us.txt","ge.us.txt",]


if __name__ == "__main__":
    data_init = pd.read_csv("Data/aapl.us.txt")
    env = TradingEnv(data_init,"aapl")

    np.random.seed(314)
    env.seed(314)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    print("SHAPE: " + str(obs_dim))

    # Option 2: deep network
    model = Sequential()
    model.add(LSTM(16, input_shape=(1,4), return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(8)) #16
    model.add(Activation('tanh'))
    model.add(Dense(4)) #16
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))
    model.add(Flatten())


    print(model.summary())


    policy = BoltzmannQPolicy() #Off-policy
    test_policy = MaxBoltzmannQPolicy() #On-policy
    #sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=1000, policy=policy)
    #sarsa.compile("adam", metrics=['mse'])
    memory = SequentialMemory(limit=50000, window_length=1)
    #TODO: Experiment with LOW GAMMA VALUES... COMPUTE LOG(gamma)/LOG(0.5) = 8.... THIS MEANS 8 steps horizon till return is 0.5x its value.
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=0.15, nb_steps_warmup=100,policy=policy, test_policy=test_policy)
    dqn.compile("adam", metrics=['mse'])

    #Training
    results = []
    np.random.shuffle(Datafiles)
    np.random.shuffle(ValidatorFiles)
    for i in range(10):
        validator = ValidatorFiles[i]
        validation_data = pd.read_csv("Data/"+validator)
        for data in Datafiles:
            if(data != validator):
                data_list = pd.read_csv("Data/"+data)
                print("Training: " + data)
                env.swap_dataset(data_list, data.split(".")[0])
                result = dqn.fit(env, nb_steps=28860, visualize=False, verbose=1)
                print("Final train funds for " + data + ": " + str(env.past_end_funds))
        print("Validating on: " + validator)
        env.swap_dataset(validation_data)
        test_res = dqn.test(env, nb_episodes=5, visualize=True)
        print("Final Funds for " + validator + ": " + str(env.past_end_funds))
        results.append(sum(test_res.history["episode_reward"])/len(test_res.history["episode_reward"])) #Average the returns for all training episodes for this cycle

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format("all"), overwrite=True)
    dqn.save
    line, = plt.plot([1,2,3,4,5,6,7,8,9,10], results, color = (random.random(),random.random(),random.random())) #Random color
    line.set_label("Validation results")
        
    plt.ylabel("Episode Reward")
    plt.xlabel("Training Cycle")
    plt.legend()
    plt.show()

#!/usr/bin/env python3

from enum import Enum
from trading_env import TradingEnv

import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import MaxBoltzmannQPolicy,BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

#Depending on python version/installed dependencies there may be issues with this module
try:
   import cPickle as pickle
except:
   import pickle


Datafiles = ["aapl.us.txt","dis.us.txt","ge.us.txt","ibm.us.txt","intc.us.txt","jpm.us.txt","msft.us.txt","nke.us.txt","v.us.txt","wmt.us.txt"]
ValidatorFiles = ["dis.us.txt","aapl.us.txt","ibm.us.txt","nke.us.txt","v.us.txt","jpm.us.txt","msft.us.txt","wmt.us.txt","intc.us.txt","ge.us.txt",]

GAMMA_VAL = 0.2 #0.2
EPS_VAL = 0.1

class TradingAgent():
    agent = None

    def __init__(self,mem_file=None,w_file=None):
        self.agent, self.env, self.memory = self.build_agent(mem_file,w_file)
        
    def build_agent(self,mem_file=None,w_file=None):
        #Create a dummy env to get size of input/output.
        #Makes it simpler if we ever choose to update env shapes.
        env = TradingEnv([],"",[])
        np.random.seed(314)
        env.seed(314)
        
        nb_actions = env.action_space.n
        obs_dim = env.observation_space.shape[0]
        model = Sequential()
        model.add(LSTM(5, input_shape=(7,4), return_sequences=True)) # 4 features + 1 bias term. 5 neurons
        model.add(Activation('tanh'))
        model.add(LSTM(4))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(4))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear')) #Best activation for BoltzmanPolicy
            

        #policy = EpsGreedyQPolicy(eps=EPS_VAL) #Off policy
        policy = BoltzmannQPolicy() #Off-policy
        test_policy = MaxBoltzmannQPolicy() #On-policy
        
        if mem_file is None:
            memory = SequentialMemory(limit=50000, window_length=7) ## returns observations of len (7,)
        else:
            (memory,memory.actions,memory.rewards, memory.terminals, memory.observations) = pickle.load(open(mem_file, "rb"))
            
        
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=GAMMA_VAL, nb_steps_warmup=100,policy=policy, test_policy=test_policy)
        dqn.compile("adam", metrics=['mse'])
        
        if w_file is not None:
            model.load_weights(w_file)

        return dqn, env, memory
        

        

if __name__ == "__main__":

    trader = TradingAgent()
    agent = trader.agent
    env = trader.env
    memory = trader.memory #Need to keep track for serialization

    #Training
    results = []
    funds_results = []
    loss = []
    np.random.shuffle(Datafiles)
    np.random.shuffle(ValidatorFiles)
    for i in range(10):
        validator = ValidatorFiles[i]
        validation_data = pd.read_csv("Data/"+validator)
        validation_pred = pd.read_csv("Data/"+validator.split(".")[0]+"_pred.csv")
        for data in Datafiles:
            if(data != validator):
                data_list = pd.read_csv("Data/"+data)
                data_pred = pd.read_csv("Data/"+data.split(".")[0]+"_pred.csv")
                print("Training: " + data)
                env.swap_dataset(data_list,data_pred,data.split(".")[0])
                result = agent.fit(env, nb_steps=1500, visualize=False, verbose=0) #Episode length is 1433 - 100 (Warmup steps) to cover historical train set
                print("Final train funds for " + data + ": " + str(env.past_end_funds[2:]) + "/Reward: " + str(result.history["episode_reward"][0]) + "/Actions: " + str(env.past_action_map))
        print("Validating on: " + validator)
        env.swap_dataset(validation_data,validation_pred,validator.split(".")[0])
        test_res = agent.test(env, nb_episodes=1, visualize=False, verbose=1)
        env.reset() #Test method doesnt call reset() after. we do this to process our results
        print("Final test funds for " + validator + ": " + str(env.past_end_funds[2:]) + "/Reward: " + str(test_res.history["episode_reward"][0]) + "/Actions: " + str(env.past_action_map))
        funds_results.append(sum(env.past_funds[2:])/len(env.past_funds[2:]))
        results.append(sum(test_res.history["episode_reward"])/len(test_res.history["episode_reward"])) #Average the returns for all training episodes for this cycle
        
    # After training is done, we save the final weights.
    agent.save_weights('weights_final.h5'.format("all"), overwrite=True)
    
    #And save memory
    mem_t = (memory,memory.actions,memory.rewards, memory.terminals, memory.observations)
    pickle.dump(mem_t, open("training_memory.dat","wb"), protocol=-1) #Binary
    
    line, = plt.plot([1,2,3,4,5,6,7,8,9,10], results, color = (random.random(),random.random(),random.random())) #Random color
    line.set_label("Validation results")
        
    plt.ylabel("Episode Reward")
    plt.xlabel("Training Cycle")
    plt.legend()
    plt.show()

    line, = plt.plot([1,2,3,4,5,6,7,8,9,10], funds_results, color = (random.random(),random.random(),random.random())) #Random color
    line.set_label("Cash")
        
    plt.ylabel("Avg Cash")
    plt.xlabel("Epoch Cycle")
    plt.legend()
    plt.show()

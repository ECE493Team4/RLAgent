from enum import Enum

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from simulation import TrainingSimulation as Simulation

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



class AgentActions(Enum):
    Buy = 0
    Sell = 1
    Hold = 2
    SellAll = 3
    
class Stock():
    def __init__(self,buy_price):
        self.buy_price = buy_price

#OpenAI Gym for Trading Simulation/Trading Agent
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    past_end_funds = []
    past_funds = []
    past_action_map = [0,0,0,0] #simple count of all actions taken, mapped by idx

    def __init__(self, dataset, data_pred, ticker):
        self.sim = Simulation(dataset, ticker, data_pred)
        
        #Current: Stock price, funds, held shares
        obs_domain_upper = np.array([
            np.finfo(np.float32).max,
            1, #Trend indicator
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ])
        
        obs_domain_lower = np.array([
            0,
            0,
            0,
            0,
        ])
        
        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Box(np.array([0,0]), np.array([10,10]), dtype=np.float32) #Max 10 buys, 10 sells.
        self.observation_space = spaces.Box(obs_domain_lower, obs_domain_upper, dtype=np.float32)
        
        self.seed()
        self.state = None
        self.last_action = None
        return
      
    #TODO: Seed
    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
      
    
    def swap_dataset(self, data, data_pred, new_name):
        self.sim = Simulation(data, new_name, data_pred)
        self.past_end_funds = []
        self.past_funds = []
        self.past_action_map = [0,0,0,0]
        self.reset()
      
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        reward = -1
        done = False
        self.last_action = action
        self.past_action_map[action] += 1
        if(action == 0): ##Buy
            reward = self.sim.buy_shares(1)
        elif(action == 1): ##Sell
            reward = self.sim.sell_shares(1)
        elif(action == 2): #Hold
            if(state[3] > 0): #Held shares
                reward = 0
            else:
                reward = -1000 #No action w/o any held shares
        else:
            if(state[3] > 0):
                reward = self.sim.sell_all()
            else:
                reward = -1000 #Sell w/o shares
        done = self.sim.step()
        #if(done):
        #    reward += self.sim.sell_all()
        self.state = self.sim.get_state()
        return self.state, reward, done, {}
      
    def reset(self):
        #Keep track of held money / money including held assets
        self.past_funds.append(self.sim.funds + self.sim.get_price() * self.sim.held_shares)
        self.past_end_funds.append(str(self.sim.funds) + "/" + str(self.sim.funds + self.sim.get_price() * self.sim.held_shares) + "/" +str(self.sim.volume_traded))
        self.sim.reset()
        self.state = self.sim.get_state()
        return self.state
      
    #Reders any visuals we desire
    def render(self, mode='human'):
        print("ACTION: " + str(self.last_action))
        print("Current State" + "(share_price:{} predicted_price:{} funds:{} held_shares:{})".format(*self.sim.get_state()) + " " + f"volume_traded:{self.sim.volume_traded} past_volumes_traded:{self.sim.past_volumes_traded}")
    
    #Close visuals
    def close(self):
        return

        
    
    

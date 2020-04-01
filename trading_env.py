from enum import Enum

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



STARTING_FUNDS = 5000
TRANSACTION_SIZE = 10
ML_MODEL_SUCC_PROB = 0.82 #NOTE: Alter this to gauge results

class AgentActions(Enum):
    Buy = 0
    Sell = 1
    Hold = 2
    
class Stock():
    def __init__(self,buy_price):
        self.buy_price = buy_price

#Holds stock data and maintains
#current time(step) in the training.
class Simulation:

    def __init__(self, data):
        self.data = data
        self.stocks_open = data["Open"]
        self.index = 0
        self.funds = STARTING_FUNDS
        self.held_shares = 0
        self.volume_traded = 0
        self.past_volumes_traded = []
        self.shares = []
        
    #Advance 1 data point (Day/hr/min)
    def step(self):
        self.index += 1
        done = (self.index == len(self.stocks_open)-1)
        return done
        
    def buy_shares(self):
        fundsBefore = self.funds
        if(self.funds - (TRANSACTION_SIZE*self.get_price()) >= 0):
            self.funds -= TRANSACTION_SIZE*self.get_price()
            self.held_shares += TRANSACTION_SIZE
            
            purchased_shares = []
            for i in range(TRANSACTION_SIZE):
                share = Stock(self.get_price())
                self.shares.append(share)
                purchased_shares.append(share)
            return 0
            #return self.calculateROI(purchased_shares, self.get_price(), "BUY")
        return -100 #Purchase W/O enough money
        
    def sell_shares(self):
        fundsBefore = self.funds
        if(self.held_shares >= TRANSACTION_SIZE):
            self.funds += TRANSACTION_SIZE*self.get_price()
            self.held_shares -= TRANSACTION_SIZE
            self.volume_traded += TRANSACTION_SIZE
            
            all_shares = list(sorted(self.shares, key=lambda stock: stock.buy_price))
            sold_shares = []
            for i in range(TRANSACTION_SIZE):
                share = all_shares[i]
                self.shares.remove(share)
                sold_shares.append(share)
            return self.calculateROI(sold_shares, self.get_price(), "SELL")
        return -100 #Sell W/O any shares
        
    def calculateROI(self, shares, current_price, type="BUY"):
        if(type == "BUY"):
           return -len(shares)
        else: #Sell
            profit = 0
            cost = 0
            for stock in shares:
                profit += (current_price - (stock.buy_price))
                cost += stock.buy_price
        return (profit/cost)

    def sell_all(self):
        fundsBefore = self.funds
        self.funds += self.held_shares * self.get_price()
        self.volume_traded += self.held_shares
        fundsAfter = self.funds
        self.held_shares = 0
        
        all_shares = list(sorted(self.shares, key=lambda stock: stock.buy_price))
        sold_shares = []
        for share in all_shares:
            self.shares.remove(share)
            sold_shares.append(share)
        
    #TODO: Return features/state of current sim for RL Agent
    def get_state(self):
        return (self.get_price(),self.get_predicted_price(), self.funds, self.held_shares)
        
    def get_price(self):
        return self.stocks_open[self.index]
    
    def get_predicted_price(self):
        if(self.index+1 < len(self.stocks_open)):
            if(np.random.random_sample(1) < ML_MODEL_SUCC_PROB):
                #return self.stocks_open[self.index+1]
                if (self.stocks_open[self.index+1] > self.get_price()):
                    return 1
                else:
                    return 0
            else: #ML is wrong
                #return self.get_price() + (self.get_price() - self.stocks_open[self.index+1]) #Add in the opposite direction of change
                if (self.stocks_open[self.index+1] > self.get_price()): #Send false result
                    return 0
                else:
                    return 1
        else:
            return self.get_price()
    
    def reset(self):
        self.index = 0
        self.funds = STARTING_FUNDS
        self.held_shares = 0
        self.past_volumes_traded.append(self.volume_traded)
        self.volume_traded = 0


#OpenAI Gym for Trading Simulation/Trading Agent
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    past_end_funds = []

    def __init__(self, dataset):
        self.sim = Simulation(dataset)
        
        #Standard stock vals only ATM
        #TODO: Add more features (Funds, ML Pred, Etc.)
        
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
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(obs_domain_lower, obs_domain_upper, dtype=np.float32)
        
        self.seed()
        self.state = None
        self.last_action = None

        return
      
    #TODO: Seed
    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
      
    
    def swap_dataset(self, data):
        self.sim = Simulation(data)
        self.past_end_funds = []
        self.reset()
      
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        reward = -1
        done = False
        self.last_action = action
        #TODO: State update code
        if(action == 0): ##Buy
            reward = self.sim.buy_shares()
        elif(action == 1): ##Sell
            reward = self.sim.sell_shares()
        else: #Hold
            if(state[3] > 0): #Held shares
                reward = 0
            else:
                reward = -100
        done = self.sim.step()
        #if(done):
        #    reward += self.sim.sell_all()
        self.state = self.sim.get_state()
        return np.array(self.state), reward, done, {}
      
    def reset(self):
        #Keep track of held money / money including held assets
        self.past_end_funds.append(str(self.sim.funds) + "/" + str(self.sim.funds + self.sim.get_price() * self.sim.held_shares) + "/" +str(self.sim.volume_traded))
        self.sim.reset()
        self.state = self.sim.get_state()
        return np.array(self.state)
      
    #Reders any visuals we desire
    def render(self, mode='human'):
        print("ACTION: " + str(self.last_action))
        print("Current State" + "(share_price:{} predicted_price:{} funds:{} held_shares:{})".format(*self.sim.get_state()) + " " + f"volume_traded:{self.sim.volume_traded} past_volumes_traded:{self.sim.past_volumes_traded}")
    
    #Close visuals
    def close(self):
        return

        
    
    

from enum import Enum
from db_controller import DBController

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
    #Advance 1 data point (Day/hr/min)
    def step(self):
        pass
    
    #Purchase shares
    def buy_shares(self):
        pass
        
    #Sell shares
    def sell_shares(self):
        pass
        
    #Calculate reward for a SELL action
    def calculateROI(self, shares, current_price):
        profit = 0
        cost = 0
        for stock in shares:
            profit += (current_price - (stock.buy_price))
            cost += stock.buy_price
        return (profit/cost)
      
    #Get observable state
    def get_state(self):
        pass
      
    #Get open price of stock
    def get_price(self):
        pass
        
    #Query and return predicted price
    def get_predicted_price(self):
        pass


    
class TrainingSimulation(Simulation):
    STARTING_FUNDS = 5000 #Training

    def __init__(self,data):
        self.data = data
        self.stocks_open = data["Open"]
        self.index = 0
        self.funds = self.STARTING_FUNDS
        self.held_shares = 0
        self.volume_traded = 0
        self.past_volumes_traded = []
        self.shares = []
        self.controller = DBController()
    
    #Advance 1 data point (Day/hr/min)
    def step(self):
        self.index += 1
        done = (self.index == len(self.stocks_open)-1)
        return done
    
    def buy_shares(self, purch_size):
        fundsBefore = self.funds
        if(self.funds - (purch_size*self.get_price()) >= 0):
            self.funds -= purch_size*self.get_price()
            self.held_shares += purch_size
            
            purchased_shares = []
            for i in range(purch_size):
                share = Stock(self.get_price())
                self.shares.append(share)
                purchased_shares.append(share)
            return 0
        return -100 #Purchase W/O enough money
    
    def sell_shares(self,purch_size):
        fundsBefore = self.funds
        if(self.held_shares >= purch_size):
            self.funds += purch_size*self.get_price()
            self.held_shares -= purch_size
            self.volume_traded += purch_size
            
            all_shares = list(sorted(self.shares, key=lambda stock: stock.buy_price))
            sold_shares = []
            for i in range(purch_size):
                share = all_shares[i]
                self.shares.remove(share)
                sold_shares.append(share)
            return super.calculateROI(sold_shares, self.get_price())
        return -100 #Sell W/O any shares
    
    def get_state(self):
        return (self.get_price(),self.get_predicted_price(), self.funds, self.held_shares)
    
    def get_price(self):
        return self.stocks_open[self.index]
        
    #Create trend indicator and return
    def get_predicted_price(self):
        pred = self.controller.get_stock_prediction()
        if (pred[0] > self.get_price()):
            return 1
        else:
            return 0
    
    def reset(self):
        self.index = 0
        self.funds = STARTING_FUNDS
        self.held_shares = 0
        self.past_volumes_traded.append(self.volume_traded)
        self.volume_traded = 0

#Simulation that serves data stepping through
#the data points used for training.
#NOTE: This sim is used specifically for the project DEMO
class HistoricalSimulation(Simulation):
    def __init__(self,controller,ticker):
        self.controller = controller
        
    #Advance 1 data point (Day/hr/min)
    def step(self):
        pass
    
    #Purchase shares
    def buy_shares(self, purch_size, user_id, user_bank, ticker, sid):
        price = self.get_price(ticker)
        if(user_bank - (purch_size*price) >= 0):
            user_bank -= purch_size*price
            self.controller.add_session_trade(price, "BUY", purch_size, sid)
            self.controller.update_user_funds(user_id, user_bank)
            return
        return
        
    #Sell
    def sell_shares(self,sell_size,held_shares,user_id,user_bank,ticker,sid):
        price = self.get_price(ticker)
        if(held_shares >= sell_size):
            user_bank += sell_size*price
            self.controller.add_session_trade(price, "SELL", sell_size, sid)
            self.controller.update_user_funds(user_id, user_bank)
            return
        return
      
    #Get observable state
    def get_state(self,sid,user):
        held_shares = self.controller.get_held_shares(sid)
        return (self.get_price(),self.get_predicted_price(), user.funds, user.held_shares)
      
    #Get open price of stock
    def get_price(self,ticker,time):
        price = self.controller.get_stock_price(ticker,time)
        return price
        
    #Query and return predicted price
    def get_predicted_price(self,ticker,time):
        pred = self.controller.get_stock_prediction()
        if (pred[0] > self.get_price()):
            return 1
        else:
            return 0

#TODO: complete
#Simulation that hooks into live datasource and queries
#Actual stock ticker data
class LiveSimulation(Simulation):
    def __init__(self,controller):
        self.controller = controller

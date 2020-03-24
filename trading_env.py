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
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents import SARSAAgent, DQNAgent
from rl.policy import MaxBoltzmannQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory



STARTING_FUNDS = 5000
TRANSACTION_SIZE = 10
ML_MODEL_SUCC_PROB = 0.82 #NOTE: Alter this to gauge results
Datafiles = ["aapl.us.txt","dis.us.txt","ge.us.txt","ibm.us.txt","intc.us.txt","jpm.us.txt","msft.us.txt","nke.us.txt","v.us.txt","wmt.us.txt"]
#Datafiles = ["aapl.us.txt"]

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
        return -1
        
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
        return -1
        
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
                return self.stocks_open[self.index+1]
            else:
                return self.get_price() + (self.get_price() - self.stocks_open[self.index+1]) #Add in the opposite direction of change
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
            np.finfo(np.float32).max,
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
        done = self.sim.step()
        #if(done):
        #    reward += self.sim.sell_all()
        self.state = self.sim.get_state()
        return np.array(self.state), reward, done, {}
      
    def reset(self):
        self.past_end_funds.append(self.sim.funds + self.sim.get_price() * self.sim.held_shares)
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

if __name__ == "__main__":
    data_init = pd.read_csv("Data/aapl.us.txt")
    env = TradingEnv(data_init)

    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    print("SHAPE: " + str(obs_dim))
    # Next, we build a very simple model.
    #model = Sequential()
    #model.add(Dense(units=16, activation='relu', input_shape=(1,3)))
    #model.add(Dense(units=8, activation='relu'))
    #model.add(Dense(nb_actions))
    #model.add(Activation('softmax'))
    #model.add(Flatten())

    # Option 1 : Simple model
    #model = Sequential()
    #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    #model.add(Dense(nb_actions))
    #model.add(Activation('softmax'))

    # Option 2: deep network
    model = Sequential()
    model.add(Dense(16, input_shape=(1,4)))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('relu'))
    model.add(Flatten())


    print(model.summary())


    policy = MaxBoltzmannQPolicy()
    #sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=1000, policy=policy)
    #sarsa.compile("adam", metrics=['mse'])
    memory = SequentialMemory(limit=28860, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=1, nb_steps_warmup=100,policy=policy)
    dqn.compile("adam", metrics=['mse'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    results = {}
    np.random.shuffle(Datafiles)
    for i in range(10):
        for data in Datafiles:
            if(data not in results):
                results[data] = []
            data_list = pd.read_csv("Data/"+data)
            print("Training: " + data)
            env.swap_dataset(data_list)
            result = dqn.fit(env, nb_steps=28860, visualize=False, verbose=1)
            print("Final Funds for " + data + ": " + str(env.past_end_funds))
            results[data].append(sum(result.history["episode_reward"])/len(result.history["episode_reward"])) #Average the returns for all training episodes for this cycle

   # results["aapl.us.txt"] = []
   # result = dqn.fit(env, nb_steps=2000, visualize=False, verbose=1)
   # results["aapl.us.txt"].append((count, result.history))

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format("all"), overwrite=True)
    
    # Finally, evaluate our algorithm for 5 episodes.
    test_res = dqn.test(env, nb_episodes=200, visualize=False)
    for res in results:
        x = results[res]
        line, = plt.plot([1,2,3,4,5,6,7,8,9,10], x, color = (random.random(),random.random(),random.random())) #Random color
        line.set_label(res)
            
        plt.ylabel("Episode Reward")
        plt.xlabel("Training Cycle")
        plt.legend()
        plt.show()
    
    
    
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    #memory = EpisodeParameterMemory(limit=1000, window_length=1)

    #cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
    #cem.compile()

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    #cem.fit(env, nb_steps=100000, visualize=False, verbose=2)

    # After training is done, we save the best weights.
    #cem.save_weights('cem_{}_params.h5f'.format("stocks-v0"), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    #cem.test(env, nb_episodes=5, visualize=True)

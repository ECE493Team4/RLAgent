from enum import Enum

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from numpy import random
import pandas as pd

import gym
# import gym_anytrading
# from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
# from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
# import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory



STARTING_FUNDS = 5000
TRANSACTION_SIZE = 1

class AgentActions(Enum):
    Buy = 0
    Sell = 1

def decision(probability) -> bool:
    return random.random() < probability


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
        fundsAfter = self.funds
        # print("perdif b{} a{} {}".format(fundsBefore, fundsAfter, (abs(fundsAfter - fundsBefore)/((fundsAfter + fundsBefore)/2))))
        return (fundsAfter - fundsBefore)/abs(fundsAfter - fundsBefore if fundsAfter - fundsBefore != 0 else 1)*(abs(fundsAfter - fundsBefore)/((fundsAfter + fundsBefore)/2))
        
    def sell_shares(self):
        fundsBefore = self.funds

        # if self.held_shares == 0:
        #     punishment = -1
        # else:
        #     punishment = 0
        if(self.held_shares >= TRANSACTION_SIZE):
            self.funds += TRANSACTION_SIZE*self.get_price()
            self.held_shares -= TRANSACTION_SIZE
            self.volume_traded += TRANSACTION_SIZE
        fundsAfter = self.funds


        return (fundsAfter - fundsBefore)/abs(fundsAfter - fundsBefore if fundsAfter - fundsBefore != 0 else 1)*(abs(fundsAfter - fundsBefore)/((fundsAfter + fundsBefore)/2))

    def sell_all(self):
        fundsBefore = self.funds
        self.funds += self.held_shares * self.get_price()
        self.volume_traded += self.held_shares
        fundsAfter = self.funds
        self.held_shares = 0
        return fundsAfter - fundsBefore

    def get_prediction_price(self):
        accuracy = 0.6
        past_price = self.get_price()
        try:
            future_price = self.stocks_open[self.index+1]
        except:
            future_price = self.get_price()

        per_dif = (abs(future_price - past_price)/((future_price + past_price)/2))
        norm_sign = (future_price - past_price)/abs(future_price - past_price if future_price - past_price != 0 else 1)
        if decision(accuracy):
            predicted_price = past_price + norm_sign * past_price * per_dif
        else:
            predicted_price = past_price - norm_sign * past_price * per_dif

        return predicted_price

    def get_price(self):
        return self.stocks_open[self.index]
        
    #TODO: Return features/state of current sim for RL Agent
    def get_state(self):
        return (self.get_price(), self.get_prediction_price(), self.funds, self.held_shares)
        
    def reset(self):
        self.index = 0
        self.funds = STARTING_FUNDS
        self.held_shares = 0
        self.past_volumes_traded.append(self.volume_traded)
        self.volume_traded = 0


#OpenAI Gym for Trading Simulation/Trading Agent
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

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
            np.finfo(np.float32).max,
            0,
            0,
            0,
        ])
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(obs_domain_lower, obs_domain_upper, dtype=np.float32)
        
        self.seed()
        self.state = None
        self.last_action = None

        return
      
    #TODO: Seed
    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
      
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        reward = 0
        done = False
        self.last_action = action
        #TODO: State update code
        if(action == 0): ##Buy
            reward = self.sim.buy_shares()
        else: ##Sell
            reward = self.sim.sell_shares()
        done = self.sim.step()
        if(done):
            reward += self.sim.sell_all()
        self.state = self.sim.get_state()
        return np.array(self.state), reward, done, {}
      
    def reset(self):
        self.sim.reset()
        self.state = self.sim.get_state()
        return np.array(self.state)
      
    #Reders any visuals we desire
    def render(self, mode='human'):
        print("ACTION: " + str(self.last_action))
        print("Current State" + "(share_price:{} predicted_price: {} funds:{} held_shares:{})".format(*self.sim.get_state()) + " " + f"volume_traded:{self.sim.volume_traded} past_volumes_traded:{self.sim.past_volumes_traded}")
    
    #Close visuals
    def close(self):
        return

if __name__ == "__main__":
    data = pd.read_csv("aapl.us.txt")
    env = TradingEnv(data)

    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    print("SHAPE: " + str(obs_dim))
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=(1,4)))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(nb_actions))
    model.add(Dense(units=2, activation='relu'))
    model.add(Activation('softmax'))
    model.add(Dense(units=2, activation='relu'))
    model.add(Flatten())

    # Option 1 : Simple model
    #model = Sequential()
    #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    #model.add(Dense(nb_actions))
    #model.add(Activation('softmax'))

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


    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.compile("adam", metrics=['mse'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    sarsa.fit(env, nb_steps=15000, visualize=False, verbose=2)
    # After training is done, we save the final weights.
    sarsa.save_weights('sarsa_{}_weights.h5f'.format("appl"), overwrite=True)
    env.sim.past_volumes_traded = []
    # Finally, evaluate our algorithm for 5 episodes.
    h = sarsa.test(env, nb_episodes=50, nb_max_start_steps=5)
    print(h.history['episode_reward'])
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

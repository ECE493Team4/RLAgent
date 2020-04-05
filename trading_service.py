import psycopg2
from enum import Enum
import pandas as pd
import pandas.io.sql as sqlio
import sys
from db_controller import DBController
from simulation import HistoricalSimulation, LiveSimulation
from trading_agent import TradingAgent

class postgresql_db_config:
    NAME = 'stock_data'
    PORT = 5432
    HOST = '162.246.156.44'
    USER = 'stock_data_admin'
    PASSWORD = 'ece493_team4_stock_data'
    
    
class AgentActions(Enum):
    Buy = 0
    Sell = 1
    Hold = 2
    SellAll = 3


class TradingService():
    
    agent = None
    time = None
    sim = None
    controller = DBController()
    
    def __init__(self, type = "HIST"):
        self.serv_type = type
        trader = TradingAgent(mem_file="training_memory_final.dat",w_file="weights_final.h5")
        self.agent = trader.agent
        if (self.serv_type == "LIVE"):
            self.sim = LiveSimulation(self.controller)
        else:
            self.sim = HistoricalSimulation(self.controller)
    
        self.poll_trading_sessions()

    def poll_trading_sessions(self):
        while(1):
            #1. Query sessions to be traded
            #2. Take action
            #3. Set trade status/user data
            #4. Busy-wait
            all_sessions = self.controller.get_active_trades()
            for session in all_sessions.iterrows():
                sid, username, ticker, num_trades = (session[1].session_id, session[1].username, session[1].ticker, session[1].num_trades)
                user = self.controller.get_user(username)
                user_funds = user.bank[0]
                user_id = user.id[0]
                held_shares = self.controller.get_held_shares(sid)
                state = self.get_state(sid, ticker, user_funds, num_trades)
                curr_price = state[0], state[2],
                action = self.take_action(state)
                t_type = None
                if action == 0:
                    self.sim.buy_shares(1,user_id,user_funds,ticker,sid,num_trades)
                elif action == 1:
                    self.sim.sell_shares(1,held_shares,user_id,user_funds,ticker,sid,num_trades)
                elif action == 3:
                    self.sim.sell_all(held_shares,user_id,user_funds,ticker,sid)
                else: #Action 2, Do nothing. This isn't recorded as it is not a Trade.
                    pass
                                
                self.controller.update_start_time(sid)
                self.controller.update_session_trades(sid,num_trades+1)
                #2. Log action to graylog, DB

    def take_action(self,state):
        #1. Take action using
        #print("TRAINING? "+str(self.agent.training)) This verifies agent is ON-POLICY in production.
        action = self.agent.forward(state)
        return action
        
    def get_state(self,sid,ticker,user_funds,num_trades):
        #1. SQL query for ML prediction
        #2. Get sim data OR Query live data
        #3. Return state (price, trend_indicator, user.funds, held_shares)
        return self.sim.get_state(sid,user_funds,ticker,num_trades)
        

if __name__ == "__main__":
    args = sys.argv
    if(len(args) > 1):
        serv_type = args[1]
    else:
        serv_type = "HIST"
    print("Launching rl agent for: "+serv_type+" config")
    serv = TradingService(serv_type)

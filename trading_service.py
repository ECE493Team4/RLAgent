import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import sys
from db_controller import DBController

class postgresql_db_config:
    NAME = 'stock_data'
    PORT = 5432
    HOST = '162.246.156.44'
    USER = 'stock_data_admin'
    PASSWORD = 'ece493_team4_stock_data'

class TradingService():
    #TODO: Simulation -> 1 for each stock?
    
    #TODO: Determine time syncing
    agent = None
    time = None
    sim = None
    controller = DBController()
    
    def __init__(self, type = "HIST"):
        self.serv_type = type
        #1. Load weights
        #2. Load memory
        #3. Load policy
        #4. init agent with params
        if (self.serv_type == "LIVE"): #TODO: Make live default during production
            self.serv_type == "LIVE"
            #self.sim = livesim()
        self.poll_trading_sessions()

    def poll_trading_sessions(self):
        while(1):
            #1. Query sessions to be traded
            #2. Take action
            #3. Set trade status/user data
            #4. Sleep? Or busy-wait
            all_sessions = self.controller.get_active_trades()
            print(all_sessions)
            for session in all_sessions:
                user = self.controller.get_user(session.username)
                state = self.get_state(session.session_id, user)
                action = self.take_action(state)
                #1. Write user/session/trades changes to DB
                #2. Log action to graylog, DB
            pass

    def take_action(self,state):
        #1. Take action
        pass
        
    def get_state(self,ticker,user):
        #1. SQL query for ML prediction
        #2. Get sim data OR Query live data
        #3. Return state (price, trend_indicator, user.funds, held_shares)
        price = self.controller.get_stock_price(ticker, self.time)
        trend_indicator = None #TODO: Query ML pred
        funds = user.bank
        shares = self.controller.get_held_shares(sid)
        return (price, trend_indicator, funds, shares)
        

if __name__ == "__main__":
    args = sys.argv
    if(len(args) > 1):
        serv_type = args[1]
    else:
        serv_type = "HIST"
    print("Launching server for: "+serv_type+" config")
    serv = TradingService(serv_type)

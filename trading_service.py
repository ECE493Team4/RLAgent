import psycopg2
import pandas as pd
import pandas.io.sql as sqlio


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
    
    def __init__(self):
        #1. Load weights
        #2. Load memory
        #3. Load policy
        #4. init agent with params
        self.poll_trading_sessions()
        pass
        
    def poll_trading_sessions(self):
        while(1):
            #1. Query sessions to be traded
            #2. Take action
            #3. Set trade status/user data
            #4. Sleep? Or busy-wait
            all_sessions = self.get_active_trades()
            for session in all_sessions:
                user = self.get_user(session.username)
                state = self.get_state(session.session_id, user)
                action = self.take_action(state)
                #1. Write user/session/trades changes to DB
                #2. Log action to graylog, DB
            pass

    def get_active_trades(self):
        sql ='''
        select * from public.trading_session where is_paused = False AND is_finished = False order by start_time asc
        '''
        db = self.get_connection()
        data = sqlio.read_sql_query(sql, db)
        db.close()
        return data
        
    def get_connection(self):
        postgre_db = psycopg2.connect(dbname = postgresql_db_config.NAME,
                                        user = postgresql_db_config.USER,
                                        password = postgresql_db_config.PASSWORD,
                                        host = postgresql_db_config.HOST,
                                        port = postgresql_db_config.PORT)
        return postgre_db

    def take_action(self,state):
        #1. Take action
        pass
        
    def get_state(self,ticker,user):
        #1. SQL query for ML prediction
        #2. Get sim data OR Query live data
        #3. Return state (price, trend_indicator, user.funds, held_shares)
        price = self.get_stock_price(ticker, self.time)
        trend_indicator = None #TODO: Query ML pred
        funds = user.bank
        shares = self.get_held_shares(sid)
        return (price, trend_indicator, funds, shares)
        
    def get_stock_price(self, ticker, time):
        return 0
        
    def get_held_shares(self, sid):
        #1. Sum BUY's
        #2. Sum SELL's
        #3. Compute and return diff
        sql ="select trade_type,price,SUM(volume) from public.trade where session_id = '"+sid+"' GROUP BY trade_type,price"
        data = sqlio.read_sql_query(sql, postgre_db)
        held_stocks = 0
        for trade in data.itertuples():
            print(trade)
            _, t_type, price, vol = trade
            if t_type == "BUY":
                held_stocks += vol
            elif t_type == "SELL":
                held_stocks -= vol
        return held_stocks
        
    def get_user(self,name):
        #1. SQL query for user data
        sql ="select * from public.user where username = '"+name+"'"
        db = self.get_connection()
        data = sqlio.read_sql_query(sql, db)
        return data


if __name__ == "__main__":
    serv = TradingService()

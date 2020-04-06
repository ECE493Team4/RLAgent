import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
from datetime import datetime, timedelta
from yahooquery import Ticker

class postgresql_db_config:
    NAME = 'stock_data'
    PORT = 5432
    HOST = '162.246.156.44'
    USER = 'stock_data_admin'
    PASSWORD = 'ece493_team4_stock_data'

class DBController():
    
    def get_connection(self):
        postgre_db = psycopg2.connect(dbname = postgresql_db_config.NAME,
                                    user = postgresql_db_config.USER,
                                    password = postgresql_db_config.PASSWORD,
                                    host = postgresql_db_config.HOST,
                                    port = postgresql_db_config.PORT)
        return postgre_db
        
    #Returns the stock price for a given ticker at a given time
    def get_stock_price(self, ticker, time):
        try:
            #Get Nth row. (Its impossible to keep date/times syncd)
            sql = "SELECT * FROM (select ROW_NUMBER() OVER (ORDER BY time_stamp asc) AS RowNum, time_stamp, open from public.stock_data_full where stock_name = '"+ticker+"') as row WHERE RowNum = '"+str(int(time+1))+"'" #NOTE: Time is an index in historical case
            db = self.get_connection()
            data = sqlio.read_sql_query(sql, db)
        except:
            print("Failed to query stock price from Datasource")
            #TODO: Graylog
            raise
        finally:
            db.close()
        return data.open
        
    #returns the current stock price for a ticker
    def get_live_stock_price(self, ticker):
        try:
            tick = Ticker(ticker)
            price = tick.price[ticker]["regularMarketPrice"]
        except:
            print("Failed to query stock price from Yahoo")
            #TODO: Graylog
            raise
        return float(price)
        
    #returns the current stock price for a ticker
    def get_live_stock_pred(self, ticker):
        try:
            #Get the most recent prediction for this stock.
            sql = "SELECT * FROM (select ROW_NUMBER() OVER (ORDER BY time_stamp desc) AS RowNum, time_stamp, prediction from public.stock_prediction where stock_name = '"+ticker+"') as row WHERE RowNum = 1" #Not 0 indexed..
            db = self.get_connection()
            data = sqlio.read_sql_query(sql, db)
        except:
            print("Failed to query stock prediction")
            #TODO: Graylog
            raise
        finally:
            db.close()
        return data.prediction
        
    #Returns the prediction for a ticker at a given time
    def get_stock_prediction(self, ticker, time):
        try:
            #Get Nth row. (Its impossible to keep date/times syncd)
            sql = "SELECT * FROM (select ROW_NUMBER() OVER (ORDER BY time_stamp asc) AS RowNum, time_stamp, prediction from public.stock_prediction where stock_name = '"+ticker+"') as row WHERE RowNum = '"+str(int(time+1))+"'" #NOTE: Time is an index in historical case
            db = self.get_connection()
            data = sqlio.read_sql_query(sql, db)
        except:
            print("Failed to query stock prediction")
            #TODO: Graylog
            raise
        finally:
            db.close()
        
        return data.prediction
    
    #Returns the number of held shares for a current session.
    def get_held_shares(self, sid):
        try:
            sql ="select trade_type,price,SUM(volume) from public.trade where session_id = '"+str(sid)+"' GROUP BY trade_type,price"
            db = self.get_connection()
            data = sqlio.read_sql_query(sql, db)
        except:
            print("failed to query held_stocks for sid: "+str(sid))
            #TODO: Graylog
            raise
        finally:
            db.close()
        
        #1. Sum BUY's
        #2. Sum SELL's
        #3. Compute and return diff
        held_stocks = 0
        for trade in data.itertuples():
            _, t_type, price, vol = trade
            if t_type == "BUY":
                held_stocks += vol
            elif t_type == "SELL":
                held_stocks -= vol
        return held_stocks

    def update_user_funds(self, uuid, new_bank):
        try:
            sql = """UPDATE public.user
            SET bank = (%s)
            WHERE id = (%s)"""
            db = self.get_connection()
            cur = db.cursor()
            cur.execute(sql, (new_bank.item(),uuid.item())) #.item() converts new_bank to native python type
            db.commit()
        except:
            print("Failed to update user funds for: "+str(uuid))
            #TODO: Graylog
            raise
        finally:
            cur.close()
            db.close()
        return
        
    def add_session_trade(self,price,trade_type,volume,session):
        try:
            sql = """INSERT INTO public.trade(price, trade_type, volume, time_stamp, session_id)
            VALUES ((%s), (%s), (%s), (%s), (%s))"""
            db = self.get_connection()
            cur = db.cursor()
            #Format date, no time-zone.
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cur.execute(sql, (price,trade_type,volume,time,session))
            db.commit()
        except:
            print("failed to add trade to session: "+str(session))
            #TODO: Graylog
            raise
        finally:
            cur.close()
            db.close()
        return
        
    def update_start_time(self, session):
        try:
            sql = """UPDATE public.trading_session SET start_time = (%s) WHERE session_id = (%s)"""
            db = self.get_connection()
            cur = db.cursor()
            next_time = (datetime.now() + timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
            cur.execute(sql, (next_time,session))
            db.commit()
        except:
            print("Failed to update start time for sid: "+str(session))
            #TODO: Graylog
            raise
        finally:
            cur.close()
            db.close()
        return
        
    #Increments session trades for a session with Sid
    def update_session_trades(self,sid,num_trades):
        try:
            sql = """UPDATE public.trading_session
            SET num_trades = (%s)
            WHERE session_id = (%s) """
            db = self.get_connection()
            cur = db.cursor()
            cur.execute(sql, (num_trades,sid))
            db.commit()
        except:
            print("Failed to update trade count for: "+str(sid))
            #TODO: Graylog
            raise
        finally:
            cur.close()
            db.close()
        return
   
   #Returns a user object
    def get_user(self,name):
        try:
            #1. SQL query for user data
            sql ="select * from public.user where username = '"+name+"'"
            db = self.get_connection()
            data = sqlio.read_sql_query(sql, db)
        except:
            print("Failed to get user: "+name)
            #TODO: Graylog
            raise
        finally:
            db.close()
        return data

    #Returns all trades that ACTIVE (Paused = false, Finished = false, start_time < Time.Now)
    #These are trades that are ready to be processed.
    def get_active_trades(self):
        try:
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sql = "select * from public.trading_session WHERE is_paused = False AND is_finished = False AND start_time < '"+str(time)+"'"
            db = self.get_connection()
            data = sqlio.read_sql_query(sql, db)
        except:
            print("Failed to get active trades")
            #TODO: Graylog
            raise
        finally:
            db.close()
        return data

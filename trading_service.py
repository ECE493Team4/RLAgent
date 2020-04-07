import argparse
import logging
import os
from logging.handlers import TimedRotatingFileHandler

import graypy as graypy
import psycopg2
from enum import Enum
import pandas as pd
import pandas.io.sql as sqlio
import sys
from db_controller import DBController
from simulation import HistoricalSimulation, LiveSimulation
from trading_agent import TradingAgent
import datetime

__log__ = logging.getLogger(__name__)


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
    
    trader = None
    agent = None
    time = None
    sim = None
    
    #Keep track of observations for each session.
    recent_obs = {}
    
    controller = DBController()
    
    def __init__(self, type = "HIST"):
        self.serv_type = type
        self.trader = TradingAgent(mem_file="training_memory_final.dat",w_file="weights_final.h5")
        self.agent = self.trader.agent
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
            ny_time = datetime.datetime.utcnow() + datetime.timedelta(hours=-4)
            ny_hour = ny_time.hour
            ny_min = ny_time.minute
            try:
                all_sessions = self.controller.get_active_trades()
            except:
                __log__.exception("failed to get active trading sessions")
                self.poll_trading_sessions() #Retry
                break
            for session in all_sessions.iterrows():
                #Only perform a trade if market is open. (Or, if this is the demo historical case)
                if(self.serv_type == "HIST" or ((ny_hour >= 10 or (ny_hour >= 9 and ny_min >= 30)) and ny_hour < 16)):
                    try:
                        sid, username, ticker, num_trades = (session[1].session_id, session[1].username, session[1].ticker, session[1].num_trades)
                        user = self.controller.get_user(username)
                        user_funds = user.bank[0]
                        user_id = user.id[0]
                        held_shares = self.controller.get_held_shares(sid)
                        state = self.get_state(sid, ticker, user_funds, num_trades)
                        curr_price = state[0], state[2],
                        action = self.take_action(state,sid)
                        t_type = None
                        if action == 0: #Buy
                            self.sim.buy_shares(1,user_id,user_funds,ticker,sid,num_trades)
                        elif action == 1: #Sell
                            self.sim.sell_shares(1,held_shares,user_id,user_funds,ticker,sid,num_trades)
                        elif action == 3: #Sell All
                            self.sim.sell_all(held_shares,user_id,user_funds,ticker,sid,num_trades)
                        else: #Action 2, Do nothing. This isn't recorded as it is not a Trade.
                            pass
                        self.controller.update_start_time(sid)
                        self.controller.update_session_trades(sid,num_trades+1)
                    except:
                        __log__.exception(f"failed to perform trade for id: {sid}")
                        #raise
                

    #Agent step and take action according to learned policy
    def take_action(self,state,sid):
        #1. Take action using
        #print("TRAINING? "+str(self.agent.training)) #This verifies agent is ON-POLICY in production.
        
        #Swap observations to this sessions past experiences
        if sid in self.recent_obs:
            self.agent.memory.recent_observations = self.recent_obs[sid]
        
        full_state = self.agent.memory.get_recent_state(state)
        action = self.agent.forward(state)
        
        # We must append recent experiences for LSTM window.
        # Note: this only adds recent experiences for LSTM window, and not to observational/reward/terminal memory
        # as training would do.
        # Since this is a non-training environment the args reward and terminal are negligable
        self.agent.memory.append(state,action,0,False)
        
        #Store recent observations
        self.recent_obs[sid] = self.agent.memory.recent_observations
        
        return action
        
    def get_state(self,sid,ticker,user_funds,num_trades):
        #1. SQL query for ML prediction
        #2. Get sim data OR Query live data
        #3. Return state (price, trend_indicator, user.funds, held_shares)
        return self.sim.get_state(sid,user_funds,ticker,num_trades)
        

LOG_LEVEL_STRINGS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def log_level(log_level_string: str):
    """Argparse type function for determining the specified logging level"""
    if log_level_string not in LOG_LEVEL_STRINGS:
        raise argparse.ArgumentTypeError(
            "invalid choice: {} (choose from {})".format(
                log_level_string, LOG_LEVEL_STRINGS
            )
        )
    return getattr(logging, log_level_string, logging.INFO)


def add_log_parser(parser):
    """Add logging options to the argument parser"""
    group = parser.add_argument_group(title="Logging")
    group.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        type=log_level,
        help="Set the logging output level",
    )
    group.add_argument(
        "--log-dir",
        dest="log_dir",
        help="Enable TimeRotatingLogging at the directory " "specified",
    )
    group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    group.add_argument(
        "--graylog-address",
        dest="graylog_address",
        help="Enable graylog for TCP log forwarding at the IP address specified.",
    )
    group.add_argument(
        "--graylog-port",
        dest="graylog_port",
        default=12201,
        help="Port for graylog TCP log forwarding.",
    )


def init_logging(args, log_file_path):
    """Intake a argparse.parse_args() object and setup python logging"""
    # configure logging
    handlers_ = []
    log_format = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] - %(message)s")
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(args.log_dir, log_file_path),
            when="d",
            interval=1,
            backupCount=7,
            encoding="UTF-8",
        )
        file_handler.setFormatter(log_format)
        file_handler.setLevel(args.log_level)
        handlers_.append(file_handler)
    if args.verbose:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(args.log_level)
        handlers_.append(stream_handler)

    if args.graylog_address:
        graylog_handler = graypy.GELFTCPHandler(args.graylog_address, args.graylog_port)
        handlers_.append(graylog_handler)

    logging.basicConfig(handlers=handlers_, level=args.log_level)


def get_parser() -> argparse.ArgumentParser:
    """Create and return the argparser for the RLAgent trading service"""
    parser = argparse.ArgumentParser(
        description="Start the RLAgent trading service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        dest="server_type",
        default="HIST",
        choices=["LIVE", "HIST"],
        help="Server type of RLAgent",
    )

    add_log_parser(parser)

    return parser


def main(argv=sys.argv[1:]) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    init_logging(args, "RLAgent.log")
    __log__.info(f"launching rl agent for: {args.server_type} config")
    serv = TradingService(args.server_type)


if __name__ == "__main__":
   main()

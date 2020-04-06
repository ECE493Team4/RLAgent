import unittest
from db_controller import DBController
from yahooquery import Ticker
#Database keys to data objects dedicated for testing
TEST_USER_ID = 0
TEST_SESSION_ID = 1
TEST_STOCK_IDX = 0
stocks = ["v","msft","dis","aapl","intc","ge","ibm","jpm","nke","wmt"]

#Test DB Controller
class TestDBController(unittest.TestCase):
    
    db = DBController()
    
    def test_get_historical_price(self):
        #Assert all tickers get historical vals
        print("Testing get historical price for all tickers..")
        for ticker in stocks:
            price = self.db.get_stock_price(ticker,TEST_STOCK_IDX)[0]
            self.assertTrue(price >= 0) #Assert price is valid.
        print("Success")

    #Testing framework will unfortunately not close sockets properly for these tests.
    def test_get_live_stock_price(self):
        import warnings #Disable resource warnings for this test
        warnings.filterwarnings("ignore", category=ResourceWarning)
        print("Testing get live price for all tickers (This may be slow)..")
        for ticker in stocks:
            tick = Ticker(ticker)
            cur_price = tick.price[ticker]["regularMarketPrice"]
            price = self.db.get_live_stock_price(ticker)
            self.assertTrue(price >= 0)
            self.assertTrue(abs(cur_price - price) <= 0.50) #Could be minor fluctuation between calls
        print("Success")

    def test_get_historical_pred(self):
        pass

    def test_get_live_stock_pred(self):
        pass

    def test_get_held_shares(self):
        shares = self.db.get_held_shares(TEST_SESSION_ID)
        self.assertTrue(shares >=0)
        pass

    def test_update_user_funds(self):
        pass
        
    def test_add_session_trade(self):
        pass
    
    def test_update_start_time(self):
        pass
    
    def test_update_session_trades(self):
        pass
        
    def test_get_user(self):
        pass
        
    def test_get_active_trades(self):
        pass


#Do testing
if __name__ == "__main__":
    unittest.main()

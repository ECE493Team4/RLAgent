import unittest
from db_controller import DBController
from yahooquery import Ticker
#Database keys to data objects dedicated for testing
TEST_USER_ID = 0
TEST_SESSION_ID = 1
TEST_NAME = "test_acc"
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
            self.assertTrue(isinstance(price, float)) #RL State price is of float
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
            self.assertTrue(isinstance(price, float)) #RL State price is of float
        print("Success")

    #Test get prediction from DB
    def test_get_historical_pred(self):
        print("Testing get historical prediction for all tickers..")
        for ticker in stocks:
            pred = self.db.get_stock_prediction(ticker,TEST_STOCK_IDX)[0][0] #Get next pred
            self.assertTrue(pred >= 0) #Assert price is valid.
            self.assertTrue(isinstance(pred, float)) #float expected
        print("Success")

    #Get live prediction
    def test_get_live_stock_pred(self):
        print("Testing get live prediction for all tickers..")
        import warnings #Disable resource warnings for this test
        warnings.filterwarnings("ignore", category=ResourceWarning)
        for ticker in stocks:
            pred = self.db.get_live_stock_pred(ticker)[0][0] #Get next pred
            price_now = self.db.get_live_stock_price(ticker)
            self.assertTrue(pred >= 0) #Assert price is valid.
            self.assertTrue(abs(price_now - pred) <= 10.00) #Cannot guarantee actual range, we just check it is close
            self.assertTrue(isinstance(pred, float)) #float expected
        print("Success")
    
    #Test get shares for a user
    def test_get_held_shares(self):
        shares = self.db.get_held_shares(TEST_SESSION_ID)
        self.assertTrue(shares >=0) #Negative shares are impossible
        self.assertTrue(isinstance(shares, int)) #RL State price is of float
        pass
    
    #Equivalence partitioning for updating funds.
    def test_update_user_funds(self):
        print("Testing add user funds")
        user = self.db.get_user(TEST_NAME)
        before_funds = user.bank[0]
        self.db.update_user_funds(user.id[0],before_funds+100)
        
        user_new = self.db.get_user(TEST_NAME)
        self.assertTrue(user_new.username[0] == TEST_NAME)
        self.assertTrue(user_new.id[0] == TEST_USER_ID)
        self.assertTrue(user_new.bank[0] == before_funds+100)
        print("Success")

        print("Testing subtract user funds")
        user = self.db.get_user(TEST_NAME)
        before_funds = user.bank[0]
        self.db.update_user_funds(user.id[0],before_funds-100)
        
        user_new = self.db.get_user(TEST_NAME)
        self.assertTrue(user_new.username[0] == TEST_NAME)
        self.assertTrue(user_new.id[0] == TEST_USER_ID)
        self.assertTrue(user_new.bank[0] == before_funds-100)
        print("Success")
        
        print("Testing add no user funds")
        user = self.db.get_user(TEST_NAME)
        before_funds = user.bank[0]
        self.db.update_user_funds(user.id[0],before_funds)
        
        user_new = self.db.get_user(TEST_NAME)
        self.assertTrue(user_new.username[0] == TEST_NAME)
        self.assertTrue(user_new.id[0] == TEST_USER_ID)
        self.assertTrue(user_new.bank[0] == before_funds)
        self.assertTrue(isinstance(before_funds, float))
        print("Success")
        pass
        
    #Equivalence tests for adding trades of different types, and verification via share count
    def test_add_session_trade(self):
        print("Testing add session trade BUY")
        shares_before = self.db.get_held_shares(TEST_SESSION_ID)
        self.db.add_session_trade(10,"BUY",1,TEST_SESSION_ID)
        
        shares_after = self.db.get_held_shares(TEST_SESSION_ID)
        self.assertTrue(shares_before < shares_after)
        self.assertTrue(shares_before+1 == shares_after)
        print("Success")

        
        print("Testing add session trade SELL")
        shares_before = self.db.get_held_shares(TEST_SESSION_ID)
        self.db.add_session_trade(10,"SELL",1,TEST_SESSION_ID)
        
        shares_after = self.db.get_held_shares(TEST_SESSION_ID)
        self.assertTrue(shares_before > shares_after)
        self.assertTrue(shares_before-1 == shares_after)
        print("Success")

        
        print("Testing add session trade Broken Trade Type")
        shares_before = self.db.get_held_shares(TEST_SESSION_ID)
        self.db.add_session_trade(10,"ABCDE",1,TEST_SESSION_ID)
        
        shares_after = self.db.get_held_shares(TEST_SESSION_ID)
        self.assertTrue(shares_before == shares_after)
        print("Success")
        
    #Test get a user from DB
    def test_get_user(self):
        print("testing get user")
        user = self.db.get_user(TEST_NAME)
        self.assertTrue(user.username[0] == TEST_NAME)
        self.assertTrue(user.id[0] == TEST_USER_ID)
        
        
    def test_get_active_trades(self):
        pass


#Do testing
if __name__ == "__main__":
    unittest.main()

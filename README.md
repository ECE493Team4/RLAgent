# RLAgent

Dependencies: 
-Python 3.6.1
-Graypy
-Psycopg2-binary
-Pandas
-Yahooquery
-Gym
-Matplotlib 
-Tensorflow==2.0.0b1 
-Keras-rl2
-Tensorboard 
-Tensorflow-estimator 

All dependencies above can be obtained using the Python pip3 installer. 

NOTE: I used a specific version of Tensorflow for Keras-RL2, which could give issues on different types of hardware or versions of Python. The above setup was used on a CentOS machine on the Cybera cloud. Please reach out if any issues are encountered due to the tensorflow version. Tensorflow may also print many warnings during initialization, but this will not necessarily cause a failure to launch.

To run the RL Agent using Historical Data for demo purposes, the application can be launched using: 

"python3 trading_service.py HIST"

This launches the Trading Service with the Historical Data as its data source for all sessions, and will peform trades on a 20-second cycle, rather than 1 hour.



To run the RL Agent using Live Data, the application can be launched using: 

"python3 trading_service.py LIVE"

This launches the Trading Service with Live Stock Data as its data source for all sessions. This service will perform trades on an hourly interval, as outlined in the requirements. 




The application will run indefinitely until stopped. 

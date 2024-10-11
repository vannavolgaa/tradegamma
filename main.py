from src.backtests.backtest2 import BacktestInput, BacktestOutput, VolatilityTraderBlacktest, load_backtest
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt
inputdata = BacktestInput()
bt = VolatilityTraderBlacktest().launch_backtest()

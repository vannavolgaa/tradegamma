from src.backtests.backtest2 import BacktestInput, BacktestOutput, VolatilityTraderBlacktest
import pandas as pd 
import matplotlib.pyplot as plt
inputdata = BacktestInput()
bt = VolatilityTraderBlacktest().launch_backtest()
bt.plot_portfolio_usd_value()
bt.plot_portfolio_delta()


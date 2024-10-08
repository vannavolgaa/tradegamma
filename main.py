import matplotlib.pyplot as plt 
from src.backtest import backtest_volatility_trader
from src.portfolio import CashFlow
from src.instruments import Currency
from src.loader import update_market_loader

update_market_loader()

test = backtest_volatility_trader()

plt.plot(test.time, test.is_cash_sufficient)
plt.show()

plt.plot(test.time, test.re_forecast)
plt.show()

plt.plot(test.time, test.delta)
plt.show()

plt.plot(test.time, test.theta)
plt.show()

plt.plot(test.time, test.initial_margin)
plt.show()

test.re_forecast


import numpy as np 
values = np.array(list(test.usd_value))
changes = np.diff(np.log(values))
sum(changes)
np.std(changes)
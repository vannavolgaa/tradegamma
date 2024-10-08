from src.backtest import backtest_volatility_trader
from src.portfolio import CashFlow
from src.instruments import Currency
import matplotlib.pyplot as plt 

test = backtest_volatility_trader()
deposit = CashFlow(100000, Currency('USD'))

plt.plot(list(test.keys()), list(test.values()))
plt.show()

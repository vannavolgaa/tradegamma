from datetime import datetime, timedelta
from src.market import MarketLoader
import matplotlib.pyplot as plt

test = MarketLoader(n=500000, do_market_processing=True)

test.get_implied_volatility_log_change_forecast(test.btcusd, max(test.dates_dt))

test.get_realized_volatility_forecast(test.btcusd, max(test.dates_dt))


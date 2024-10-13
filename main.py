from datetime import datetime, timedelta
from src.trader.voltrader import BacktestVolatilityTraderInput, BacktestVolatilityTrader

inputdata = BacktestVolatilityTraderInput(
    deposit_usd  = 100000,
    reload_market  = False ,
    first_data_point  = 0,
    last_data_point  = 1000000, 
    exposure=0.1,
    perpetual_mark_price_dt=timedelta(days=10),
    atm_factor_dt=timedelta(days=10)
)
bt = BacktestVolatilityTrader(inputdata).launch_backtest()

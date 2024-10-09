from src.backtest import BacktestInput, BacktestOutput, VolatilityTraderBlacktest

inputdata = BacktestInput()
bt = VolatilityTraderBlacktest().launch_backtest()

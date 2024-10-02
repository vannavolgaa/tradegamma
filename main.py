from src.loader import MarketLoader
from src.strategy import StrategyExecution, StrategyPnL

loader = MarketLoader()
bt_dates = loader.get_date_vector_for_backtest()
reftime = bt_dates[0]
risk_factor = loader.btcusd
market = [m for m in loader.markets 
           if m.reference_time==reftime 
           and m.risk_factor==risk_factor][0]
iv_change = loader.get_implied_volatility_log_change_forecast(risk_factor,reftime)
revol = loader.get_realized_volatility_forecast(risk_factor,reftime)
stratexec = StrategyExecution(market,0.1)
len(stratexec.generate_calendar_spread_block_trades())
len(stratexec.generate_straddle_block_trades())
blocktrade = stratexec.find_winning_block_trades(iv_change,revol)
best_blocktrade = stratexec.find_best_block_trade(iv_change,revol)
for bt in blocktrade:
    print('----')
    print(bt.trades[0].instrument.name)
    print(bt.trades[1].instrument.name)
    print(bt.trades[0].number_contracts)
    print(bt.trades[1].number_contracts)
    stratpnl = StrategyPnL(bt,market,iv_change,revol,False)
    print(stratpnl.estimated_pnl())



best_blocktrade.trades[0].instrument.name
best_blocktrade.trades[1].instrument.name
StrategyPnL(best_blocktrade,market,iv_change,revol,False).estimated_pnl()

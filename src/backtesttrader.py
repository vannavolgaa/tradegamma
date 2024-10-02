from src.loader import MarketLoader
from src.trader import CashFlow
from src.instruments import (
    Currency
)

class BacktestTrader: 
    def __init__(self) -> None:
        self.market = MarketLoader()
        self.date_vector = self.market.get_date_vector_for_backtest()
        self.trades = list()
    
    def initial_deposit(self) -> CashFlow: 
        return CashFlow(10000, Currency('USD'))
    
    
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from src.loader import get_market_loader, MarketLoader, update_market_loader
from src.traders.trader1 import VolatilityTrader, Portfolio, CashFlow, Trade, Position, VTBook, Currency
from src.instruments import Option, Spot, PerpetualFuture

def backtest_volatility_trader() -> pd.DataFrame: 
    loader = get_market_loader()
    deposit = CashFlow(100000, loader.usd)
    exposure = 0.1
    dates = loader.get_date_vector_for_backtest()
    date0 = dates[0]
    risk_factor = loader.btcusd
    market = [m for m in loader.markets 
          if m.risk_factor==risk_factor
          and m.reference_time==date0][0]
    spot_instrument = market.spot 
    spot_quote = market.get_quote(spot_instrument.name)
    quantity_to_trade = deposit.amount/spot_quote.order_book.best_ask
    initial_trade = Trade(spot_instrument, quantity_to_trade,
                        spot_quote.order_book.best_ask,
                        loader.usd,date0)
    book = VTBook(list(), [initial_trade])
    output = list()
    i = 0
    for d in dates: 
        print(d)
        iv_forecast = loader.get_implied_volatility_log_change_forecast(
            risk_factor, d)
        re_forecast = loader.get_realized_volatility_forecast(risk_factor,d)
        market = [m for m in loader.markets 
          if m.risk_factor==risk_factor
          and m.reference_time==d][0]
        trader = VolatilityTrader(
            book, market,exposure,iv_forecast,re_forecast,deposit)
        book = trader.update_book()
        pft = book.to_portfolio(market)
        margin = pft.get_margin()
        data = {
            'time' : d, 
            'iv_change_forecast' : iv_forecast, 
            're_forecast' : re_forecast, 
            'usd_value' : pft.get_usd_value(deposit), 
            'initial_margin': margin.initial, 
            'maintenance_margin' : margin.maintenance, 
            'delta' : pft.sensitivities.delta, 
            'gamma' : pft.sensitivities.gamma, 
            'vega' : pft.sensitivities.vega, 
            'theta' : pft.sensitivities.theta, 
            'usd_realised': pft.get_usd_realised_pnl(deposit), 
            'usd_unrealised' : pft.get_usd_unrealised_pnl(), 
            'is_cash_sufficient' : pft.is_cash_sufficient(deposit), 
            'total_fee_usd' : pft.get_usd_fee_value()
        }
        output.append(data)
        i = i +1
        perc = round(100*i/len(dates),2)
        print(str(perc) + '% of the backtest')
    print('Backtest is done.')
    return pd.DataFrame(output)

@dataclass
class BacktestInput: 
    deposit_usd : float = 10000
    exposure : float = 0.1
    reload_market : bool = False 
    first_data_point : int = 0 
    last_data_point : int = 1000000

    def get_market_loader(self) -> MarketLoader: 
        if self.reload_market: 
            update_market_loader(self.first_data_point,self.last_data_point)
        return get_market_loader()  

@dataclass
class BacktestOutput: 
    portfolio_report : pd.DataFrame
    option_position_report : pd.DataFrame
    perp_position_report : pd.DataFrame
    last_book : VTBook
    last_portfolio : Portfolio

    @staticmethod
    def x_axis__dates(dates: List[datetime]):
        dates = [d.to_pydatetime() for d in dates]
        now, then = dates[0], dates[len(dates)-1]
        return mdates.drange(now,then,timedelta(hours=24))

    def plot_portfolio_delta(self) -> None: 
        plt.plot(self.portfolio_report.time, 
                 self.portfolio_report.delta)
        plt.title('Global delta exposure')
        plt.show()
    
    def plot_portfolio_usd_value(self) -> None: 
        plt.plot(self.portfolio_report.time, 
                 self.portfolio_report.usd_value)
        plt.title('Global USD portfolio value')
        plt.show()

@dataclass
class VolatilityTraderBlacktest: 
    def __init__(self, inputdata : BacktestInput = BacktestInput()): 
        self.inputdata = inputdata
        self.loader = inputdata.get_market_loader()
        self.deposit = CashFlow(inputdata.deposit_usd,self.loader.usd)
        self.risk_factor = self.loader.btcusd
        self.dates = self.loader.get_date_vector_for_backtest()
        self.forecasts = self.get_forecasts()

    def get_forecasts(self) -> dict[datetime,dict[str, float]]:
        output = dict()
        for d in self.dates: 
            iv_forecast = self.loader.get_implied_volatility_log_change_forecast(
                self.risk_factor, d)
            re_forecast = self.loader.get_realized_volatility_forecast(
                self.risk_factor,d)
            output[d] = {'iv_change_forecast':iv_forecast, 
                         're_forecast' : re_forecast}
        return output 

    def initialize_book(self) -> VTBook: 
        date0 = self.dates[0]
        market = [m for m in self.loader.markets 
          if m.risk_factor==self.risk_factor
          and m.reference_time==date0][0]
        spot_instrument = market.spot 
        spot_quote = market.get_quote(spot_instrument.name)
        quantity_to_trade = self.deposit.amount/spot_quote.order_book.best_ask
        initial_trade = Trade(spot_instrument, quantity_to_trade,
                            spot_quote.order_book.best_ask,
                            self.loader.usd,date0)
        return VTBook(list(), [initial_trade])

    def update_book(self, reference_time:datetime, initial_book:VTBook) -> VTBook: 
        market = [m for m in self.loader.markets 
          if m.risk_factor==self.risk_factor
          and m.reference_time==reference_time][0]
        trader = VolatilityTrader(
            book=initial_book, 
            market=market,
            exposure=self.inputdata.exposure,
            forecast_iv_change=self.forecasts[reference_time]['iv_change_forecast'],
            forecast_re=self.forecasts[reference_time]['re_forecast'],
            initial_deposit= self.deposit)
        return trader.update_book()
    
    def portfolio_report(self, portfolio:Portfolio) -> dict: 
        ref_date = portfolio.market.reference_time
        margin = portfolio.get_margin()
        spot = portfolio.spot_quote.order_book.mid
        n_open_position = len([p for p in portfolio.positions if p.number_contracts!=0])
        return {
            'time' : ref_date, 
            'usd_value' : portfolio.get_usd_value(self.deposit), 
            'initial_margin_usd': margin.initial*spot, 
            'maintenance_margin_usd' : margin.maintenance*spot, 
            'delta' : portfolio.sensitivities.delta, 
            'gamma' : portfolio.sensitivities.gamma, 
            'vega' : portfolio.sensitivities.vega, 
            'theta' : portfolio.sensitivities.theta, 
            'usd_realised': portfolio.get_usd_realised_pnl(self.deposit), 
            'usd_unrealised' : portfolio.get_usd_unrealised_pnl(), 
            'is_cash_sufficient' : portfolio.is_cash_sufficient(self.deposit), 
            'total_fee_usd' : portfolio.get_usd_fee_value(), 
            'number_open_position': n_open_position, 
            'iv_change_forecast' : self.forecasts[ref_date]['iv_change_forecast'], 
            're_forecast' : self.forecasts[ref_date]['re_forecast']
        }

    def option_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        iv_change_forecast = self.forecasts[market.reference_time]['iv_change_forecast']
        re_forecast = self.forecasts[market.reference_time]['re_forecast']
        spot = portfolio.spot_quote.order_book.mid
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, Option): 
                i = p.instrument
                quote = market.get_quote(i.name)
                dict_out = {
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price,
                    't' : i.time_to_expiry(market.reference_time), 
                    'strike' : i.strike, 
                    'iv' : quote.mid_iv,
                    'iv_forecast' : quote.mid_iv*(1+iv_change_forecast), 
                    're_forecast' : re_forecast,  
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p), 
                    'usd_realised': p.get_realised_pnl_cash_flow().amount*spot, 
                    'usd_fees': p.get_fee_cash_flow().amount*spot     
                }
                output.append(dict_out)
            else: continue
        return output
    
    def perp_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, PerpetualFuture): 
                i = p.instrument
                quote = market.get_quote(i.name)
                dict_out = {
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price, 
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p),    
                }
                output.append(dict_out)
            else: continue
        return output
        
    def launch_backtest(self) -> BacktestOutput: 
        i = 0
        book = self.initialize_book()
        portfolio_report = list()
        option_position_report = list()
        perp_position_report = list()
        for d in self.dates: 
            print(d)
            book = self.update_book(d, book)
            market = [m for m in self.loader.markets 
                    if m.risk_factor==self.risk_factor
                    and m.reference_time==d][0]
            pft = book.to_portfolio(market)
            portfolio_report.append(self.portfolio_report(pft))
            option_position_report = option_position_report + self.option_position_report(pft)
            perp_position_report = perp_position_report + self.perp_position_report(pft)
            i = i +1
            perc = round(100*i/len(self.dates),2)
            print(str(perc) + '% of the backtest')
        print('Backtest is done.')
        return BacktestOutput(
            portfolio_report=pd.DataFrame(portfolio_report), 
            option_position_report=pd.DataFrame(option_position_report), 
            perp_position_report=pd.DataFrame(perp_position_report), 
            last_book=book,
            last_portfolio=pft)
        

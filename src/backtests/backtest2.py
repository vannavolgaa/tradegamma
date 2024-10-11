from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import pickle 
import numpy as np 
from src.loader import get_market_loader, MarketLoader, update_market_loader
from src.traders.trader2 import VolatilityTrader, Portfolio, CashFlow, Trade, Position, VTBook, Currency
from src.instruments import Option, Spot, PerpetualFuture
from src.quant.blackscholes import BlackScholes

@dataclass
class BacktestInput: 
    deposit_usd : float = 100000
    pnl_target : float = 0.0001
    reload_market : bool = False 
    first_data_point : int = 1000000
    last_data_point : int = 2000000

    def get_market_loader(self) -> MarketLoader: 
        if self.reload_market: 
            update_market_loader(self.first_data_point,self.last_data_point)
        return get_market_loader()  

@dataclass
class BacktestOutput: 
    portfolio_report : pd.DataFrame
    option_position_report : pd.DataFrame
    perp_position_report : pd.DataFrame
    spot_position_report : pd.DataFrame
    market_data_report : pd.DataFrame
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
    
    def plot_portfolio_total_usd_pnl(self) -> None: 
        plt.plot(self.portfolio_report.time, 
                 self.portfolio_report.total_usd_pnl)
        plt.title('Global USD portfolio value')
        plt.show()
    
    def plot_portfolio_gamma(self) -> None: 
        plt.plot(self.portfolio_report.time, 
                 self.portfolio_report.gamma)
        plt.title('Global gamma exposure')
        plt.show()
    
    def plot_portfolio_vega(self) -> None: 
        plt.plot(self.portfolio_report.time, 
                 self.portfolio_report.vega)
        plt.title('Global vega exposure')
        plt.show()
    
    def plot_delta_hedge_efficiency(self) -> None: 
        pass 

    def date_filter_report(self, dates: List[datetime], data:pd.DataFrame): 
        data_dt = [t for t in list(data.time) if t.to_pydatetime() in dates]
        return data[data.time.isin(data_dt)]
    
    
def save_backtest(data:BacktestOutput) -> None: 
    tmsp = round(datetime.now().timestamp())
    path = 'data/backtest/'+str(tmsp)+'_backtest.pkl'
    with open(path, 'wb') as outp:  
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def load_backtest(backtest_timestamp:int) -> BacktestOutput: 
    path = 'data/backtest/'+str(backtest_timestamp)+'_backtest.pkl'
    with open(path, 'rb') as inp:
        return pickle.load(inp)

@dataclass
class VolatilityTraderBlacktest: 
    def __init__(self, inputdata : BacktestInput = BacktestInput()): 
        self.inputdata = inputdata
        self.loader = inputdata.get_market_loader()
        self.deposit = CashFlow(inputdata.deposit_usd,self.loader.usd)
        self.risk_factor = self.loader.btcusd
        self.dates = self.loader.get_date_vector_for_backtest()[0:100]
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
        return VTBook([initial_trade], self.deposit)

    def update_book(self, reference_time:datetime, initial_book:VTBook) -> VTBook: 
        market = [m for m in self.loader.markets 
          if m.risk_factor==self.risk_factor
          and m.reference_time==reference_time][0]
        trader = VolatilityTrader(
            book=initial_book, 
            market=market,
            target_pnl=self.inputdata.pnl_target,
            forecast_iv_change=self.forecasts[reference_time]['iv_change_forecast'],
            forecast_re=self.forecasts[reference_time]['re_forecast'])
        return trader.update_book()
    
    def portfolio_report(self, portfolio:Portfolio) -> dict: 
        ref_date = portfolio.market.reference_time
        margin = portfolio.get_margin()
        spot = portfolio.spot_quote.order_book.mid
        n_open_position = len([p for p in portfolio.positions if p.number_contracts!=0])
        return {
            'time' : ref_date, 
            'total_usd_pnl' : portfolio.get_usd_total_pnl(), 
            'realised_usd_pnl': portfolio.get_usd_realised_pnl(), 
            'unrealised_usd_pnl' : portfolio.get_usd_unrealised_pnl(), 
            'initial_margin_usd': margin.initial*spot, 
            'maintenance_margin_usd' : margin.maintenance*spot, 
            'delta' : portfolio.sensitivities.delta, 
            'gamma' : portfolio.sensitivities.gamma, 
            'vega' : portfolio.sensitivities.vega, 
            'theta' : portfolio.sensitivities.theta, 
            'is_cash_sufficient' : portfolio.is_cash_sufficient(), 
            'total_fee_usd' : portfolio.get_usd_fee(), 
            'number_open_position': n_open_position, 
            'iv_change_forecast' : self.forecasts[ref_date]['iv_change_forecast'], 
            're_forecast' : self.forecasts[ref_date]['re_forecast'], 

        }

    def option_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        iv_change_forecast = self.forecasts[market.reference_time]['iv_change_forecast']
        re_forecast = self.forecasts[market.reference_time]['re_forecast']
        spot = portfolio.spot_quote.order_book.mid
        futts = market.get_future_term_structure()
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, Option): 
                i = p.instrument
                quote = market.get_quote(i.name)
                sigma = market.get_implied_volatility_quote(i.name,'mid')
                t = i.time_to_expiry(market.reference_time)
                F = futts.future_price(t)
                if i.call_or_put == 'C': put_or_call=1
                else: put_or_call=-1
                bs = BlackScholes(
                    S = F, K = i.strike, t = t, sigma = sigma, 
                    q = 0, r=0, future=0, call_or_put=put_or_call)
                q = p.number_contracts*i.contract_size
                dict_out = {
                    'time': market.reference_time,
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price,
                    'delta' : q*bs.delta(),
                    'gamma' : q*bs.gamma(),
                    'vega' : q*bs.vega(),
                    'theta' : q*bs.theta(),
                    't' : i.time_to_expiry(market.reference_time), 
                    'strike' : i.strike, 
                    'iv' : quote.mid_iv,
                    'iv_forecast' : quote.mid_iv*(1+iv_change_forecast), 
                    're_forecast' : re_forecast,  
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p), 
                    'usd_realised': p.get_realised_pnl()*spot, 
                    'usd_fees': p.get_fee_cash_flow().amount*spot     
                }
                output.append(dict_out)
            else: continue
        return output
    
    def perp_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        spot = portfolio.spot_quote.order_book.mid
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, PerpetualFuture): 
                i = p.instrument
                quote = market.get_quote(i.name)
                dict_out = {
                    'time': market.reference_time,
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price, 
                    'usd_realised': p.get_realised_pnl()*spot,
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p),    
                }
                output.append(dict_out)
            else: continue
        return output
    
    def spot_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, Spot): 
                i = p.instrument
                quote = market.get_quote(i.name)
                dict_out = {
                    'time': market.reference_time,
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price, 
                    'usd_realised': p.get_realised_pnl(),
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p),    
                }
                output.append(dict_out)
            else: continue
        return output
    
    def market_data_report(self, ref_time:datetime) -> List[dict]: 
        market =  [m for m in self.loader.markets 
                    if m.risk_factor==self.risk_factor
                    and m.reference_time==ref_time][0]
        deltat = timedelta(hours=1)
        next_date = ref_time + deltat 
        market_next =  [m for m in self.loader.markets 
                    if m.risk_factor==self.risk_factor
                    and m.reference_time==next_date][0]
        perp_quote = market.get_quote(market.perpetual.name).order_book.mark_price
        perp_quote_next = market_next.get_quote(market_next.perpetual.name).order_book.mark_price
        atm_factor = market.atm_factor
        atm_factor_next = market_next.atm_factor
        perp_log_change = np.log(perp_quote_next)-np.log(perp_quote)
        atmf_log_change = np.log(atm_factor_next)-np.log(atm_factor)
        dt = 1/(365*24)
        perp_rev = np.sqrt((perp_log_change**2)/dt)
        re_forecat = self.forecasts[ref_time]['re_forecast']
        iv_change_forecast = self.forecasts[ref_time]['iv_change_forecast']
        return {
            'time' : ref_time,
            'perp_log_return' : perp_log_change, 
            'perp_realised_vol': perp_rev, 
            'atm_factor_change' : atmf_log_change, 
            'atm_factor_change_forecast_1' : iv_change_forecast, 
            'perp_realised_vol_forecast_1': re_forecat}

    def launch_backtest(self) -> BacktestOutput: 
        i = 0
        book = self.initialize_book()
        portfolio_report = list()
        option_position_report = list()
        perp_position_report = list()
        spot_position_report = list()
        market_data_report = list()
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
            spot_position_report = spot_position_report + self.spot_position_report(pft)
            market_data_report.append(self.market_data_report(d))
            i = i +1
            perc = round(100*i/len(self.dates),2)
            print(str(perc) + '% of the backtest')
        print('Backtest is done.')
        output = BacktestOutput(
            portfolio_report=pd.DataFrame(portfolio_report), 
            option_position_report=pd.DataFrame(option_position_report), 
            perp_position_report=pd.DataFrame(perp_position_report), 
            spot_position_report=pd.DataFrame(spot_position_report),
            market_data_report = pd.DataFrame(market_data_report),
            last_book=book,
            last_portfolio=pft)
        save_backtest(output)
        return output 
        

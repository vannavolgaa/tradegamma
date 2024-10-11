from typing import List 
from dataclasses import dataclass
import numpy as np 
from datetime import datetime
from src.portfolio import (
    Trade, 
    CashFlow, 
    Portfolio, 
    Position, 
    trades_to_positions, 
    settle_expired_position
)
from src.instruments import Option, Currency, Future
from src.market import Market 

@dataclass
class VTBook: 
    trades : List[Trade]
    initial_deposit : CashFlow

    def to_positions(self) -> List[Position]: 
        return trades_to_positions(self.trades)

    def to_portfolio(self, market:Market) -> Portfolio: 
        return Portfolio(self.to_positions(),market, self.initial_deposit)

def settle_book_expired_positions(book:VTBook, market:Market) -> VTBook: 
    new_position = [settle_expired_position(p, market) for p in book.to_positions()]
    trades = list()
    for p in new_position: 
        trades = trades + p.trades
    return VTBook(trades, book.initial_deposit)

@dataclass
class VTEnginePnL: 
    portfolio : Portfolio
    forecast_iv_change : float 
    forecast_re : float
    add_fee : float = 0

    def __post_init__(self):
        self.market = self.portfolio.market
        self.spot_quote = self.market.get_quote(self.market.spot.name)
        self.dt = 1/(365*24)
        self.futts = self.market.get_future_term_structure()
        self._set_data()

    def _set_data(self) -> None: 
        self.sigma = list()
        self.gamma = list()
        self.vega = list()
        self.F = list()
        for p in self.portfolio.positions: 
            if p.number_contracts == 0: continue
            pftbis = Portfolio([p], self.market, CashFlow(0, Currency('USD')))
            i = p.instrument
            if not isinstance(i,Option): continue
            name = p.instrument.name
            if p.number_contracts>=0: 
                sigma=self.market.get_implied_volatility_quote(name,'ask')
            else: 
                sigma=self.market.get_implied_volatility_quote(name,'bid')
            t = i.time_to_expiry(self.market.reference_time) 
            self.F.append(self.futts.future_price(t))
            self.sigma.append(sigma)
            self.gamma.append(pftbis.sensitivities.gamma)
            self.vega.append(pftbis.sensitivities.vega)

    def gamma_pnl(self) -> float:
        s, g, F = np.array(self.sigma), np.array(self.gamma), np.array(self.F)
        dreiv = self.forecast_re - s
        dreiv_factor = dreiv*(.5*dreiv + s)
        return np.sum(g*(F**2)*self.dt*dreiv_factor)
    
    def vega_pnl(self) -> float: 
        vega = np.array(self.portfolio.sensitivities.vega)
        change_sigma = self.forecast_iv_change*np.array(self.sigma)
        return np.sum(vega*change_sigma)
    
    def delta_hedge_fee(self) -> float: 
        trade = self.portfolio.perpetual_delta_hedging_trade()
        fee = trade.trade_fee()
        spot_price = self.spot_quote.order_book.mid
        if fee.currency == Currency('USD'): return fee.amount
        else: return spot_price*fee.amount
    
    def estimated_pnl(self) -> float: 
        fee = self.delta_hedge_fee()+self.add_fee
        return self.gamma_pnl()+self.vega_pnl()-fee

@dataclass
class VolatilityTradingStrategy: 
    market : Market 
    forecast_iv_change : float 
    forecast_re : float
    pnl_target : float = 0.02

    def __post_init__(self): 
        self.fic, self.fre = self.forecast_iv_change, self.forecast_re
        self.atm_chain = self.market.atm_chain
        self.block_trades = self.generate_cs_block_trades()\
            +self.generate_sdle_block_trades()
        spot_quote = self.market.get_quote(self.market.spot.name)
        self.spot = spot_quote.order_book.mid
        
    def opt_blocktrade_generator(
            self, 
            first_leg: Option, 
            second_leg: Option, 
            same_direction: bool) -> List[List[Trade]]: 
        fl_quote = self.atm_chain.mapped_quotes[first_leg.name]
        sl_quote = self.atm_chain.mapped_quotes[second_leg.name]
        reftime = self.market.reference_time
        firstdata = {'instrument':first_leg, 
                     'reference_time': reftime, 
                     'currency':fl_quote.order_book.quote_currency}
        seconddata = {'instrument':second_leg, 
                     'reference_time': reftime, 
                     'currency':sl_quote.order_book.quote_currency}
        fdatalong, fdatashort = firstdata.copy(), firstdata.copy()
        sdatalong, sdatashort = seconddata.copy(), seconddata.copy()
        fdatalong['number_contracts']=first_leg.minimum_contract_amount
        sdatalong['number_contracts']=second_leg.minimum_contract_amount
        fdatashort['number_contracts']=-first_leg.minimum_contract_amount
        sdatashort['number_contracts']=-second_leg.minimum_contract_amount
        fdatalong['traded_price'] = fl_quote.order_book.best_ask
        sdatalong['traded_price'] = sl_quote.order_book.best_ask
        fdatashort['traded_price'] = fl_quote.order_book.best_bid
        sdatashort['traded_price'] = sl_quote.order_book.best_bid
        flongtrade = Trade(**fdatalong)
        fshorttrade = Trade(**fdatashort)
        slongtrade = Trade(**sdatalong)
        sshorttrade = Trade(**sdatashort)
        if same_direction: 
            return [[flongtrade, slongtrade],[fshorttrade, sshorttrade]]
        else: 
            return [[flongtrade, sshorttrade], [fshorttrade, slongtrade]]

    def generate_cs_block_trades(self) -> List[List[Trade]]: 
        output = list()
        for i in range(0, len(self.atm_chain.calls)): 
             c1 = self.atm_chain.calls[i]
             for u in range(i+1, len(self.atm_chain.calls)): 
                 c2 = self.atm_chain.calls[u]
                 output = output+self.opt_blocktrade_generator(c1,c2,False)
        for i in range(0, len(self.atm_chain.puts)): 
             p1 = self.atm_chain.puts[i]
             for u in range(i+1, len(self.atm_chain.puts)): 
                 p2 = self.atm_chain.puts[u]
                 output = output+self.opt_blocktrade_generator(p1,p2,False)
        return output 

    def generate_sdle_block_trades(self) -> List[List[Trade]]: 
        output = list()
        for c in self.atm_chain.calls: 
            for p in self.atm_chain.puts: 
                output = output + self.opt_blocktrade_generator(c,p,True)
        return output  
    
    def get_fees(self, trades: List[Trade]) -> float: 
        feecf = [t.trade_fee() for t in trades]
        fees = 0
        for f in feecf:
            if f.currency!=Currency('USD'): 
                fees = fees + f.amount*self.spot
            else: fees = fees + f.amount
        return fees

    def get_block_trade_pnl(self, trades: List[Trade]) -> float: 
        pos = trades_to_positions(trades)
        pft = Portfolio(pos, self.market, CashFlow(0,Currency('USD')))
        fee = self.get_fees(trades)
        engine = VTEnginePnL(pft,self.fic,self.fre,fee)
        return engine.estimated_pnl()

    def get_best_block_trade(self) -> List[Trade]: 
        estimated_pnl = [self.get_block_trade_pnl(bt)
                         for bt in self.block_trades]
        best_bt = [bt for bt,p in zip(self.block_trades, estimated_pnl) 
                   if p == max(estimated_pnl)
                   and p>=0]
        if len(best_bt) == 0: return list()
        else: return best_bt[0]

@dataclass
class VolatilityTrader: 
    book : VTBook
    market : Market
    target_pnl : float 
    forecast_iv_change : float 
    forecast_re : float

    def __post_init__(self): 
        self.book = settle_book_expired_positions(self.book, self.market)
        self.volstrat = VolatilityTradingStrategy(
            market = self.market, 
            forecast_iv_change = self.forecast_iv_change, 
            forecast_re = self.forecast_re)
        self.new_trades = self.volstrat.get_best_block_trade()
        self.portfolio = self.book.to_portfolio(self.market)
    
    def get_bet_size(self) -> float: 
        if len(self.new_trades)==0: return 0
        target_pnl = self.portfolio.get_usd_total_value()*self.target_pnl
        pft_engine = VTEnginePnL(
            self.portfolio,
            self.forecast_iv_change,
            self.forecast_re,0)
        pft_pnl = pft_engine.estimated_pnl()
        bt_pnl = self.volstrat.get_block_trade_pnl(self.new_trades)
        if pft_pnl>=target_pnl: return 0
        return (target_pnl-pft_pnl)/bt_pnl
    
    def get_trades(self) -> List[Trade]: 
        size = self.get_bet_size()
        if size == 0: return list()
        number_contracts = list()
        prices = list()
        for b in self.new_trades: 
            n = size * b.number_contracts
            quote = self.market.get_quote(b.instrument.name)
            if n<0: 
                number_contracts.append(min(n,quote.order_book.best_bid_size))
                prices.append(quote.order_book.best_bid)
            else: 
                number_contracts.append(min(n,quote.order_book.best_ask_size))
                prices.append(quote.order_book.best_ask)
        ncontract_to_trade = min(number_contracts)
        trades = list()
        for b,p in zip(self.new_trades,prices): 
            trade_dict = b.__dict__
            trade_dict['traded_price'] = p
            trade_dict['number_contracts'] = ncontract_to_trade
            trades.append(Trade(**trade_dict))
        return trades

    def new_trade_execution(self, trades:List[Trade]) -> List[Trade]: 
        all_trades = self.book.trades + trades
        new_book = VTBook(all_trades, self.book.initial_deposit)
        new_portfolio = new_book.to_portfolio(self.market)
        delta_hedge_trade = new_portfolio.perpetual_delta_hedging_trade()
        all_trades.append(delta_hedge_trade)
        final_book = VTBook(all_trades, self.book.initial_deposit)
        final_pft = final_book.to_portfolio(self.market)
        if final_pft.is_cash_sufficient(): 
            trades.append(delta_hedge_trade)
            return trades
        else: return list() 

    def update_book(self) -> VTBook: 
        trades = self.new_trade_execution(self.get_trades())
        if len(trades) == 0: 
            trades = [self.portfolio.perpetual_delta_hedging_trade()]
        all_trades = self.book.trades + trades
        return VTBook(all_trades, self.book.initial_deposit)

    

    






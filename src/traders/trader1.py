from typing import List 
from dataclasses import dataclass
import numpy as np 
from datetime import datetime
from src.portfolio import (
    Trade, 
    CashFlow, 
    Portfolio, 
    Position, 
    trades_to_positions
)
from src.instruments import Option, Currency
from src.market import Market 

@dataclass
class VTBlockTrade: 
    _id : str
    trades : List[Trade] 
    is_block_trade_close : bool = False

    def block_trade_cost(self) -> CashFlow: 
        fees = [t.trade_fee() for t in self.trades]
        fee = sum([f.amount for f in fees])
        return CashFlow(amount = fee, currency=fees[0].currency)
    
    def to_positions(self) -> List[Position]: 
        return trades_to_positions(self.trades)
    
    def to_portfolio(self, market:Market) -> Portfolio: 
        return Portfolio(self.to_positions(), market)
    
@dataclass
class VTBlockTradePnL: 
    block_trade : VTBlockTrade
    market : Market 
    to_close : bool 
    forecast_iv_change : float 
    forecast_re : float
    load_market_data : bool = True

    def __post_init__(self):
        self.spot_quote = self.market.get_quote(self.market.spot.name)
        self.dt = 1/(365*24)
        if self.load_market_data:
            self.portfolio = self.block_trade.to_portfolio(self.market)
            self.sigma = self.get_implied_volatilities()
            futts = self.market.get_future_term_structure()
            t_vec = [t.instrument.time_to_expiry(self.market.reference_time) 
                    for t in self.block_trade.trades]
            self.F = np.array([futts.future_price(tt) for tt in t_vec]) 

    def get_implied_volatilities(self) -> np.array: 
        iv = list()
        for t in self.block_trade.trades: 
            name = t.instrument.name
            if self.to_close: 
                if t.number_contracts>=0: 
                    sigma=self.market.get_implied_volatility_quote(name,'bid')
                else: 
                    sigma=self.market.get_implied_volatility_quote(name,'ask')
            else: 
                if t.number_contracts>=0: 
                    sigma=self.market.get_implied_volatility_quote(name,'ask')
                else: 
                    sigma=self.market.get_implied_volatility_quote(name,'bid')
            iv.append(sigma)
        return np.array(iv)

    def gamma_pnl(self) -> float:
        dreiv = self.forecast_re - self.sigma
        dreiv_factor = dreiv*(.5*dreiv + self.sigma)
        gamma = self.portfolio.sensitivities.gamma
        return np.sum(gamma*(self.F**2)*self.dt*dreiv_factor)
    
    def vega_pnl(self) -> float: 
        vega = self.portfolio.sensitivities.vega
        change_sigma = self.forecast_iv_change*self.sigma
        return np.sum(vega*change_sigma)
    
    def fee(self) -> float: 
        fee_list = list()
        spot_price = self.spot_quote.order_book.mid
        for p in self.block_trade.to_positions(): 
            fee = p.get_fee_cash_flow()
            if fee.currency == Currency('USD'): 
                fee_list.append(fee.amount)
            else: 
                fee_list.append(spot_price*fee.amount)
        return sum(fee_list) 
    
    def delta_hedge_fee(self) -> float: 
        trade = self.portfolio.perpetual_delta_hedging_trade()
        fee = trade.trade_fee()
        spot_price = self.spot_quote.order_book.mid
        if fee.currency == Currency('USD'): return fee.amount
        else: return spot_price*fee.amount
    
    def estimated_pnl(self) -> float: 
        fee = self.delta_hedge_fee()+self.fee()
        return self.gamma_pnl()+self.vega_pnl()-fee

    def closing_block_trades(self) -> List[Trade]: 
        trade_list = list()
        for t in self.block_trade.trades: 
            quote = self.market.get_quote(t.instrument.name)
            if t.number_contracts>0: price = quote.order_book.best_bid 
            else: price = quote.order_book.best_ask 
            trade = Trade(
                t.instrument, 
                number_contracts=-t.number_contracts,
                traded_price=price,
                currency=t.currency, 
                reference_time=self.market.reference_time)
            trade_list.append(trade)
        return trade_list

    def settle_block_trade(self) -> VTBlockTrade: 
        ref_time = self.market.reference_time
        option_is_settle = [True for t in self.block_trade.trades
                            if t.instrument.expiry_dt==ref_time]
        S = self.spot_quote.order_book.mid
        if sum(option_is_settle)>0: 
            trade_list = list()
            for t in self.block_trade.trades: 
                trade_list.append(t)
                if t.instrument.expiry_dt == ref_time: 
                    i = t.instrument
                    K = i.strike
                    if i.call_or_put=='C': x = 1 
                    else: x=-1
                    price = x*max(S-K, 0)/S
                    trade = Trade(
                        instrument=i, 
                        number_contracts=-t.number_contracts,
                        traded_price=price,
                        currency=t.currency, 
                        reference_time=ref_time) 
                    trade_list.append(trade) 
                else:
                    quote = self.market.get_quote(t.instrument.name)
                    if t.number_contracts>0: price = quote.order_book.best_bid 
                    else: price = quote.order_book.best_ask 
                    trade = Trade(
                        instrument=t.instrument, 
                        number_contracts=-t.number_contracts,
                        traded_price=price,
                        currency=t.currency, 
                        reference_time=ref_time) 
                    trade_list.append(trade)
            return VTBlockTrade(self.block_trade._id,trade_list,True)
        else: 
            return self.block_trade

    def update_block_trade(self) -> VTBlockTrade: 
        if self.estimated_pnl()>0: return self.block_trade 
        else: 
            trades = self.closing_block_trades()+self.block_trade.trades
            return VTBlockTrade(self.block_trade._id,trades,True)

class VolatilityTradingStrategy: 
    def __init__(self, market: Market, exposure:float): 
        self.exposure = exposure
        self.market = market 
        self.atm_chain = self.market.atm_chain
        self.block_trades = self.generate_cs_block_trades()\
            +self.generate_sdle_block_trades()
        
    def opt_blocktrade_generator(
            self, 
            first_leg: Option, 
            second_leg: Option, 
            same_direction: bool) -> List[VTBlockTrade]: 
        exp = self.exposure
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
        fdatalong['number_contracts']=min(fl_quote.order_book.best_ask_size,exp)
        sdatalong['number_contracts']=min(sl_quote.order_book.best_ask_size,exp)
        fdatashort['number_contracts']=-min(fl_quote.order_book.best_bid_size,exp)
        sdatashort['number_contracts']=-min(sl_quote.order_book.best_bid_size,exp)
        fdatalong['traded_price'] = fl_quote.order_book.best_ask
        sdatalong['traded_price'] = sl_quote.order_book.best_ask
        fdatashort['traded_price'] = fl_quote.order_book.best_bid
        sdatashort['traded_price'] = sl_quote.order_book.best_bid
        flongtrade = Trade(**fdatalong)
        fshorttrade = Trade(**fdatashort)
        slongtrade = Trade(**sdatalong)
        sshorttrade = Trade(**sdatashort)
        if same_direction: 
            _id1 = 'LONG_'+first_leg.name+'_LONG_'+second_leg.name
            _id2 = 'SHORT_'+first_leg.name+'_SHORT_'+second_leg.name
            bt1 = VTBlockTrade(_id1,[flongtrade, slongtrade], False)
            bt2 = VTBlockTrade(_id2,[fshorttrade, sshorttrade], False)
        else: 
            _id1 = 'LONG_'+first_leg.name+'_SHORT_'+second_leg.name
            _id2 = 'SHORT_'+first_leg.name+'_LONG_'+second_leg.name
            bt1 = VTBlockTrade(_id1,[flongtrade, sshorttrade], False)
            bt2 = VTBlockTrade(_id2,[fshorttrade, slongtrade], False)
        return [bt1, bt2]

    def generate_cs_block_trades(self) -> List[VTBlockTrade]: 
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

    def generate_sdle_block_trades(self) -> List[VTBlockTrade]: 
        output = list()
        for c in self.atm_chain.calls: 
            for p in self.atm_chain.puts: 
                output = output + self.opt_blocktrade_generator(c,p,True)
        return output  
    
    def get_best_block_trade(
            self, 
            forecast_iv_change : float,
            forecast_re : float) -> VTBlockTrade: 
        estimated_pnl = list()
        for bt in self.block_trades: 
            strat = VTBlockTradePnL(
                bt,self.market,False,forecast_iv_change,forecast_re)
            estimated_pnl.append(strat.estimated_pnl())
        best_bt = [bt for bt,p in zip(self.block_trades, estimated_pnl) 
                   if p == max(estimated_pnl)
                   and p>=0]
        if len(best_bt) == 0: return None
        else: return best_bt[0]

@dataclass
class VTBook: 
    vt_block_trades : List[VTBlockTrade]
    delta_hedging_trades : List[Trade] 

    def to_positions(self) -> List[Position]: 
        trades = list()
        for bt in self.vt_block_trades: 
            trades = trades + bt.trades
        trades = trades + self.delta_hedging_trades
        return trades_to_positions(trades)
    
    def to_portfolio(self, market:Market) -> Portfolio: 
        return Portfolio(self.to_positions(), market)

@dataclass
class VolatilityTrader: 
    book : VTBook
    market : Market 
    exposure : float 
    forecast_iv_change : float 
    forecast_re : float
    initial_deposit : CashFlow

    def __post_init__(self): 
        self.book = self.initial_book_update()
        self.bt_ids = [bt._id for bt in self.book.vt_block_trades]

    def initial_book_update(self) -> VTBook: 
        new_bt = list()
        for bt in self.book.vt_block_trades: 
            if not bt.is_block_trade_close: 
                vt_pnl = VTBlockTradePnL(
                    block_trade=bt, 
                    market=self.market, 
                    to_close=True,
                    forecast_iv_change=np.nan, 
                    forecast_re=np.nan, 
                    load_market_data=False)
                new_bt.append(vt_pnl.settle_block_trade())
            else: 
                new_bt.append(bt)  
        return VTBook(new_bt,self.book.delta_hedging_trades)
    
    def check_new_block_trade(self, bt:VTBlockTrade) -> List[VTBlockTrade]: 
        if bt is None: return list()
        else: 
            if bt._id in self.bt_ids:
                corresponding_bt = [b for b in self.book.vt_block_trades 
                                    if b._id == bt._id][0]
                if corresponding_bt.is_block_trade_close: return [bt]
                else: return list()
            else: return [bt]

    def update_block_trades(self, find_new_bt:bool) -> List[VTBlockTrade]: 
        fivc, fre = self.forecast_iv_change, self.forecast_re
        if find_new_bt:
            voltradingstrat = VolatilityTradingStrategy(self.market,self.exposure)
            best_bt = voltradingstrat.get_best_block_trade(fivc, fre)
        else: best_bt = None
        if best_bt is None: output = list()
        else: output = [best_bt]
        #output = self.check_new_block_trade(best_bt)
        for bt in self.book.vt_block_trades: 
            if not bt.is_block_trade_close: 
                vt_pnl = VTBlockTradePnL(
                    block_trade=bt, 
                    market=self.market, 
                    to_close=True,
                    forecast_iv_change=fivc, 
                    forecast_re=fre)
                output.append(vt_pnl.update_block_trade())
            else: output.append(bt)  
        return output
    
    def update_delta_hedge_trades(self, bts: List[VTBlockTrade]) -> List[Trade]: 
        trades = self.book.delta_hedging_trades
        book = VTBook(bts, trades)
        portfolio = book.to_portfolio(self.market)
        delta_hedge_trade = portfolio.perpetual_delta_hedging_trade()
        trades.append(delta_hedge_trade)
        return trades
    
    def check_new_portfolio(self) -> bool: 
        ubt = self.update_block_trades(True)
        book = VTBook(ubt, self.update_delta_hedge_trades(ubt))
        portfolio = book.to_portfolio(self.market)
        return portfolio.check_cash(self.initial_deposit)
    
    def update_book(self) -> VTBook: 
        ubt = self.update_block_trades(True)
        return VTBook(ubt, self.update_delta_hedge_trades(ubt))



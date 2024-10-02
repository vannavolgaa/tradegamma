from dataclasses import dataclass
from typing import List
from datetime import datetime
import numpy as np 
from src.loader import MarketLoader, Market
from src.trader import Trade, CashFlow
from src.instruments import InstrumentQuote, Currency, RiskFactor, Option
from src.quant.blackscholes import BlackScholes


@dataclass
class BlockTrade: 
    trades : List[Trade] 

    def block_trade_cost(self) -> CashFlow: 
        fees = [t.trade_fee() for t in self.trades]
        fee = sum([f.amount for f in fees])
        return CashFlow(amount = fee, currency=fees[0].currency)

@dataclass
class StrategyPnL: 
    option_block_trade : BlockTrade
    market : Market
    forecast_iv_change : float 
    forecast_re : float
    to_close : bool
    
    def __post_init__(self): 
        self.dt = 1/(365*24)
        self.reference_time = self.market.reference_time
        self.risk_factor = self.market.risk_factor
        self.bs = self.get_black_scholes()
        self.gamma = self.bs.gamma()
        self.vega = self.bs.vega()
        self.delta = self.bs.delta()
        self.n = self.get_quantities()

    def delta_hedging_cost(self) -> float: 
        perp = self.market.perpetual
        quote = self.market.get_quote(perp.name)
        exposure = np.sum(np.array(self.delta)*np.array(self.n))
        spot_quote = self.market.get_quote(self.market.spot.name)
        spot_mid = spot_quote.order_book.mid
        quantity = exposure*spot_mid/perp.contract_size
        if exposure<=0: price = quote.order_book.best_ask
        else: price = quote.order_book.best_bid
        trade = Trade(
            instrument = perp,
            number_contracts=quantity,
            traded_price=price,
            marked_price=quote.order_book.mark_price,
            currency=quote.order_book.quote_currency, 
            trade_date_time=self.reference_time
        ) 
        fee = trade.trade_fee()
        if fee.currency == Currency('USD'): 
            return fee.amount
        else: 
            spot_quote = self.market.get_quote(self.market.spot.name)
            return fee.amount*spot_mid


    def block_trade_cost(self) -> float: 
        fee = self.option_block_trade.block_trade_cost()
        if fee.currency == Currency('USD'): 
            return fee.amount
        else: 
            spot_quote = self.market.get_quote(self.market.spot.name)
            return fee.amount*spot_quote.order_book.mid
    
    def get_implied_volatility(self) -> List[float]: 
        vol = list()
        for t in self.option_block_trade.trades: 
            quote = self.market.get_quote(t.instrument.name)
            if self.to_close: 
                if t.number_contracts>=0: 
                    vol.append(quote.bid_iv/100) 
                else: 
                    vol.append(quote.ask_iv/100)  
            else: 
                if t.number_contracts>=0: 
                    vol.append(quote.ask_iv/100)   
                else: 
                    vol.append(quote.bid_iv/100) 
        return vol  
    
    def get_strikes(self) -> List[float]: 
        return [t.instrument.strike 
                for t in self.option_block_trade.trades]
    
    def get_time_to_expiry(self) -> List[float]: 
        return [t.instrument.time_to_expiry(self.reference_time) 
                for t in self.option_block_trade.trades]
    
    def get_future_price(self) -> List[float]: 
        t_vec = self.get_time_to_expiry()
        futts = self.market.get_future_term_structure()
        return [futts.future_price(t) for t in t_vec]
    
    def get_putcall_indicator(self) -> List[float]: 
        return [1 if t.instrument.call_or_put=='C' else -1 
                for t in self.option_block_trade.trades]
    
    def get_black_scholes(self) -> BlackScholes: 
        return BlackScholes(
            S = np.array(self.get_future_price()), 
            K = np.array(self.get_strikes()), 
            t = np.array(self.get_time_to_expiry()), 
            sigma = np.array(self.get_implied_volatility()), 
            q = 0, r=0, future=0, 
            call_or_put=np.array(self.get_putcall_indicator()))
    
    def get_quantities(self) -> List[float]: 
        return [t.instrument.contract_size*t.number_contracts 
                for t in self.option_block_trade.trades]
    
    def estimated_gamma_pnl(self) -> float: 
        iv = np.array(self.get_implied_volatility())
        dreiv = self.forecast_re - iv
        dreiv_factor = dreiv*(.5*dreiv + iv)
        fut = np.array(self.get_future_price())
        q = np.array(self.get_quantities())
        return np.sum(q*self.gamma*(fut**2)*self.dt*dreiv_factor)

    def estimated_vega_pnl(self) -> float: 
        q = np.array(self.get_quantities())
        div = self.forecast_iv_change*np.array(self.get_implied_volatility())
        return np.sum(q*self.vega*div)
    
    def estimated_pnl(self) -> float: 
        fee = self.block_trade_cost()+self.delta_hedging_cost()
        return self.estimated_gamma_pnl()+self.estimated_vega_pnl()-fee

class StrategyExecution: 
    def __init__(self, market: Market, exposure:float): 
        self.exposure = exposure
        self.market = market 
        self.atm_chain = self.market.atm_chain
    
    def generate_block_trades(
            self, 
            first_leg: Option, 
            second_leg: Option, 
            same_direction: bool) -> List[BlockTrade]: 
        exp = self.exposure
        fl_quote = self.atm_chain.mapped_quotes[first_leg.name]
        sl_quote = self.atm_chain.mapped_quotes[second_leg.name]
        reftime = self.market.reference_time
        fmark = fl_quote.order_book.mark_price
        smark = fl_quote.order_book.mark_price
        firstdata = {'instrument':first_leg, 
                     'marked_price': fmark, 
                     'trade_date_time': reftime, 
                     'currency':fl_quote.order_book.quote_currency}
        seconddata = {'instrument':second_leg, 
                     'marked_price': smark, 
                     'trade_date_time': reftime, 
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
            bt1 = BlockTrade([flongtrade, slongtrade])
            bt2 = BlockTrade([fshorttrade, sshorttrade])
        else: 
            bt1 = BlockTrade([flongtrade, sshorttrade])
            bt2 = BlockTrade([fshorttrade, slongtrade])
        return [bt1, bt2]
    
    def generate_calendar_spread_block_trades(self) -> List[BlockTrade]: 
        output = list()
        for i in range(0, len(self.atm_chain.calls)): 
             call1 = self.atm_chain.calls[i]
             for u in range(i+1, len(self.atm_chain.calls)): 
                 call2 = self.atm_chain.calls[u]
                 output = output+self.generate_block_trades(call1,call2,False)
        for i in range(0, len(self.atm_chain.puts)): 
             put1 = self.atm_chain.puts[i]
             for u in range(i+1, len(self.atm_chain.puts)): 
                 put2 = self.atm_chain.puts[u]
                 output = output+self.generate_block_trades(put1,put2,False)
        return output 

    def generate_straddle_block_trades(self) -> List[BlockTrade]: 
        output = list()
        for c in self.atm_chain.calls: 
            for p in self.atm_chain.puts: 
                output = output + self.generate_block_trades(c,p,True)
        return output  
    
    def find_best_block_trade(
            self, 
            forecast_iv_change : float,
            forecast_re : float) -> BlockTrade: 
        all_block_trades = self.generate_calendar_spread_block_trades() + \
        self.generate_straddle_block_trades()
        estimated_pnl = list()
        for bt in all_block_trades: 
            strat = StrategyPnL(
                bt,self.market,forecast_iv_change,forecast_re, False)
            estimated_pnl.append(strat.estimated_pnl())
        best_bt = [bt for bt,p in zip(all_block_trades, estimated_pnl) 
                   if p == max(estimated_pnl)
                   and p>=0]
        if len(best_bt) == 0: return None
        else: return best_bt[0]
    
    def find_winning_block_trades(
            self, 
            forecast_iv_change : float,
            forecast_re : float) -> BlockTrade: 
        all_block_trades = self.generate_calendar_spread_block_trades() + \
        self.generate_straddle_block_trades()
        estimated_pnl = list()
        for bt in all_block_trades: 
            strat = StrategyPnL(
                bt,self.market,forecast_iv_change,forecast_re, False)
            estimated_pnl.append(strat.estimated_pnl())
        return [bt for bt,p in zip(all_block_trades, estimated_pnl) 
                   if p>=0]
        

    




    

    

    

    

    



    
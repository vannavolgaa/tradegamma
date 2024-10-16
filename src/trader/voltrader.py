from typing import List 
from dataclasses import dataclass
import numpy as np 
from datetime import timedelta, datetime
from src.portfolio import (
    Trade, 
    CashFlow, 
    Book, 
    Portfolio)
from src.instruments import Option, Currency
from src.market import Market
from src.quant.timeserie import TimeSerie
from src.trader.base import (
    TraderParameters, 
    Trader, 
    BacktestInput, 
    BacktestTrader, 
    OptionPnlEngine)

@dataclass
class VolatilityBlockTrade: 
    trades : List[Trade]
    market : Market

    def __post_init__(self): 
        self._id = self.get_id()
        self.book =  Book(self.trades, CashFlow(0,Currency('USD')))
        self.pnlengine = OptionPnlEngine(self.book,self.market)
        self.spotq = self.market.get_quote(self.market.spot.name)

    def get_id(self) -> str: 
        _id = ''
        for t in self.trades: 
            if t.number_contracts<0: 
                _id = _id + '_SHORT_'+t.instrument.name
            elif t.number_contracts>0: 
                _id = _id + '_LONG_'+t.instrument.name
            else: 
                _id = _id + '_NEUTRAL_'+t.instrument.name
        return _id
    
    def get_trades_usd_fee(self) -> float: 
        fee = 0
        for t in self.trades: 
            tfee = t.trade_fee()
            if tfee.currency!=Currency('USD'):
                fee = fee + tfee.amount*self.spotq.order_book.mid
            else: fee = fee + tfee.amount
        return fee
    
    def net_fee_usd_volatility_pnl(
            self, 
            dt:float, 
            iv_relative_change:float, 
            realised_volatility:float) -> float: 
        theta_gamma_appprox = self.pnlengine.proxy_theta_gamma_pnl(
            realised_volatility,dt) 
        vega=0#vega = self.pnlengine.vega_pnl(iv_relative_change)
        fee = 0 #self.pnlengine.delta_hedge_fee() #+self.get_trades_usd_fee()
        return theta_gamma_appprox+vega+fee

def generate_block_trade(leg1:Option, leg2:Option, 
                         same_direction: bool, market:Market, 
                         exposure: float) -> List[VolatilityBlockTrade]: 
    atm_chain = market.atm_chain
    quote1 = atm_chain.mapped_quotes[leg1.name]
    quote2 = atm_chain.mapped_quotes[leg2.name]
    firstdata = {'instrument':leg1, 
                    'reference_time': market.reference_time, 
                    'currency':quote1.order_book.quote_currency}
    seconddata = {'instrument':leg2, 
                    'reference_time':  market.reference_time, 
                    'currency':quote2.order_book.quote_currency}
    fdatalong, fdatashort = firstdata.copy(), firstdata.copy()
    sdatalong, sdatashort = seconddata.copy(), seconddata.copy()
    fdatalong['number_contracts']=min(quote1.order_book.best_ask_size,exposure)
    sdatalong['number_contracts']=min(quote2.order_book.best_ask_size,exposure)
    fdatashort['number_contracts']=-min(quote1.order_book.best_bid_size,exposure)
    sdatashort['number_contracts']=-min(quote2.order_book.best_bid_size,exposure)
    fdatalong['traded_price'] = quote1.order_book.best_ask
    sdatalong['traded_price'] = quote2.order_book.best_ask
    fdatashort['traded_price'] = quote1.order_book.best_bid
    sdatashort['traded_price'] = quote2.order_book.best_bid
    flongtrade = Trade(**fdatalong)
    fshorttrade = Trade(**fdatashort)
    slongtrade = Trade(**sdatalong)
    sshorttrade = Trade(**sdatashort) 
    if same_direction: 
        bt1 = VolatilityBlockTrade([flongtrade, slongtrade], market)
        bt2 = VolatilityBlockTrade([fshorttrade, sshorttrade], market)
    else: 
        bt1 = VolatilityBlockTrade([flongtrade, sshorttrade], market)
        bt2 = VolatilityBlockTrade([fshorttrade, slongtrade], market)
    return [bt1, bt2]

def get_atm_calendar_spread(market: Market, exposure: float, call=True) -> List[VolatilityBlockTrade]: 
    atm_chain = market.atm_chain
    output = list()
    if call: options = atm_chain.calls
    else: options = atm_chain.puts
    mapped_cs = dict()
    for i in range(0, len(options)-1): 
        for u in range(0, len(options)-1): 
            n1, n2 = options[i].name, options[u].name
            keys = list(mapped_cs.keys())
            combi1, combi2 = n1+n2, n2+n1
            if n1 == n2: continue
            elif combi1 in keys or combi2 in keys : continue
            else: 
                mapped_cs[n1+n2] = [options[i], options[u]]
    for k in list(mapped_cs.keys()):
        opt1, opt2 = mapped_cs[k][0], mapped_cs[k][1]
        output = output + generate_block_trade(
            opt1, opt2, False, market, exposure)
    return output

def get_atm_straddle(market: Market, exposure: float) -> List[VolatilityBlockTrade]:
    output = list()
    for c in market.atm_chain.calls: 
        for p in market.atm_chain.puts: 
            output = output + generate_block_trade(
                c, p, True,market, exposure)
    return output 

def get_atm_block_trades(market: Market, exposure: float) -> List[VolatilityBlockTrade]: 
    return get_atm_straddle(market,exposure)\
        +get_atm_calendar_spread(market, exposure,True)\
        +get_atm_calendar_spread(market, exposure,False)

@dataclass
class VolatilityTraderParameters(TraderParameters): 
    exposure : float 
    perpetual_mark_price_ts : TimeSerie
    atm_factor_ts : TimeSerie
    dt : float = 1/(365*24)
    stop_loss : float = -0.2
    stop_profit : float = 1
    max_spread_to_close : float = 0.1

class VolatilityTrader(Trader): 
    def __init__(self, parameters : VolatilityTraderParameters): 
        self.parameters = parameters
        self.market = self.parameters.market
        self.book = self.parameters.book
        self.iv_rchange_forecast = self.get_iv_relative_change_forecast()
        self.revol_forecast = self.get_realised_vol_forecast()

    def get_iv_relative_change_forecast(self) -> float: 
        ts = self.parameters.atm_factor_ts
        ar = ts.ar_1lag_fit()
        ivlc = ar.params['Const'].item()
        lr = ts.log_difference
        n = len(lr)
        r,a = lr[len(lr)-1], ar.params['y[1]'].item()
        ivlc = ivlc + r*a
        #for i in range(1,2): 
        #    name = 'y['+str(i)+']'
        #    r,a = lr[len(lr)-i], ar.params[name].item()
        #    ivlc = ivlc + r*a
        return ivlc.item() 

    def get_realised_vol_forecast(self) -> float: 
        dt = self.parameters.dt
        ts = self.parameters.perpetual_mark_price_ts
        egarch = ts.normal_garch_fit()
        params = egarch.params
        res, condsigmas = egarch.resid, egarch.conditional_volatility
        e, s = res[len(res)-1], condsigmas[len(condsigmas)-1]
        omega = params['omega'].item()
        alpha = params['alpha[1]'].item()
        beta = params['beta[1]'].item()
        variance = omega + alpha*np.abs(e)**2 + beta*s**2
        vol = np.sqrt(1/dt)*np.sqrt(variance)
        return vol.item() 

    def get_trades(self) -> List[Trade]: 
        bt = get_atm_block_trades(self.parameters.market, self.parameters.exposure)
        mapped_pnl = dict()
        for b in bt: 
            mapped_pnl[b._id] = b.net_fee_usd_volatility_pnl(
                self.parameters.dt,
                self.iv_rchange_forecast, 
                self.revol_forecast
            )
        mapped_pnl = dict(sorted(mapped_pnl.items(), 
                                 key=lambda item: item[1],
                                 reverse=True))
        winner_id = list(mapped_pnl.keys())[0]
        if mapped_pnl[winner_id]>0:
            wbt = [b for b in bt if b._id==winner_id][0]
            return wbt.trades 
        else: return list()
    
    def close_existing_trades(self) -> List[Trade]: 
        output = list()
        for p in self.book.to_positions(): 
            output = output + p.trades
            if p.number_contracts!=0 and isinstance(p.instrument, Option):
                bt = VolatilityBlockTrade(p.trades, self.market)
                pnl = bt.net_fee_usd_volatility_pnl(
                    self.parameters.dt,
                    self.iv_rchange_forecast, 
                    self.revol_forecast
                )
                fee = bt.get_trades_usd_fee()
                quote = self.market.get_quote(p.instrument.name) 
                markprice = quote.order_book.mark_price
                if p.number_contracts<0: 
                    price = quote.order_book.best_ask 
                else: price = quote.order_book.best_bid 
                n = p.number_contracts*p.instrument.contract_size
                pft = Portfolio([p], self.market, self.book.initial_deposit)
                if pnl < 0 and pft.get_usd_total_pnl()>0: 
                    quote = self.market.get_quote(p.instrument.name) 
                    if p.number_contracts<0: price = quote.order_book.best_ask 
                    else: price = quote.order_book.best_bid 
                    if p.instrument.name == 'BTC-3FEB23-24000-P': print(price)
                    trade = p.get_settlement_trade(price, self.market.reference_time)
                    output.append(trade)
                else: continue
        return output
    
    def close_existing_trades2(self) -> list[Trade]: 
        output = list()
        for p in self.book.to_positions(): 
            output = output + p.trades
            if p.number_contracts!=0 and isinstance(p.instrument, Option):
                quote = self.market.get_quote(p.instrument.name) 
                bt = VolatilityBlockTrade(p.trades, self.market)
                pnl = bt.net_fee_usd_volatility_pnl(
                    self.parameters.dt,
                    self.iv_rchange_forecast, 
                    self.revol_forecast
                )
                fifo = p.fifo_price
                mark = quote.order_book.mark_price
                bid = quote.order_book.best_bid 
                ask = quote.order_book.best_ask
                perf = np.sign(p.number_contracts)*(mark-fifo)/fifo
                try: bid_spread = (bid-mark)/mark
                except ZeroDivisionError as e: bid_spread = -np.inf
                try: ask_spread = (ask-mark)/mark
                except ZeroDivisionError as e: ask_spread = np.inf
                if pnl>0: 
                    cond_sp = perf<self.parameters.stop_profit
                    if cond_sp: continue
                    else: 
                        if np.sign(p.number_contracts)==-1: 
                            if ask_spread>self.parameters.max_spread_to_close:
                                continue 
                            else: 
                                trade = p.get_settlement_trade(
                                    ask, self.market.reference_time) 
                                output.append(trade)
                        else: 
                            if bid_spread<-self.parameters.max_spread_to_close:
                                continue 
                            else: 
                                trade = p.get_settlement_trade(
                                    bid, self.market.reference_time) 
                                output.append(trade)  
                else: 
                    #cond_sl = perf>self.parameters.stop_loss
                    #cond_sp = perf<self.parameters.stop_profit
                    #if cond_sl and cond_sp: continue
                    #else: 
                    if np.sign(p.number_contracts)==-1: 
                        if ask_spread>self.parameters.max_spread_to_close:
                            continue 
                        else: 
                            trade = p.get_settlement_trade(
                                ask, self.market.reference_time) 
                            output.append(trade)
                    else: 
                        if bid_spread<-self.parameters.max_spread_to_close:
                            continue 
                        else: 
                            trade = p.get_settlement_trade(
                                bid, self.market.reference_time) 
                            output.append(trade)  
        return output

    def update_book(self) -> Book:
        trades = self.get_trades()
        book_trades = self.close_existing_trades2()
        updated_trades = trades+book_trades
        updated_book = Book(updated_trades, self.book.initial_deposit)
        pft = updated_book.to_portfolio(self.market)
        delta_hedging_trade = pft.perpetual_delta_hedging_trade()
        updated_trades.append(delta_hedging_trade)
        updated_book = Book(updated_trades, self.book.initial_deposit)
        pft = updated_book.to_portfolio(self.market)
        if pft.is_cash_sufficient(): return updated_book
        else: 
            updated_book = Book(book_trades, self.book.initial_deposit)
            pft = updated_book.to_portfolio(self.market)
            delta_hedging_trade = pft.perpetual_delta_hedging_trade()
            book_trades.append(delta_hedging_trade)
            return Book(book_trades, self.book.initial_deposit)

@dataclass
class BacktestVolatilityTraderInput(BacktestInput): 
    exposure : float = 0.1
    perpetual_mark_price_dt : timedelta = timedelta(days=30)
    atm_factor_dt : timedelta = timedelta(days=30)
    dt : float = 1/(365*24)
    stop_loss : float = -0.2
    stop_profit : float = 1
    max_spread_to_close : float = 0.1

class BacktestVolatilityTrader(BacktestTrader): 
    def __init__(self, inputdata: BacktestVolatilityTraderInput):
        self.parameters = inputdata
        super().__init__(inputdata)
    
    
    def get_date_vector_for_backtest(self, dates:List[datetime]) -> List[datetime]: 
        min_date = min(dates)
        perp_dt = self.parameters.perpetual_mark_price_dt
        atmf_dt = self.parameters.atm_factor_dt
        max_dt = max([perp_dt,atmf_dt]) 
        min_date_for_bt = min_date + max_dt
        return [d for d in dates if d>min_date_for_bt]
    
    def update_book(self, reference_time: datetime, initial_book: Book) -> Book:
        market = [m for m in self.loader.markets 
          if m.risk_factor==self.risk_factor
          and m.reference_time==reference_time][0]
        perpetual_mark_price_ts = self.loader.get_instrument_mark_price_time_serie(
            market.perpetual.name, reference_time, 
            self.parameters.perpetual_mark_price_dt)
        atm_factor_ts = self.loader.get_risk_factor_atm_factor_time_serie(
            self.risk_factor, reference_time, 
            self.parameters.atm_factor_dt)
        traderparam = VolatilityTraderParameters(
            book = initial_book,
            market=market,
            exposure=self.parameters.exposure,
            dt = self.parameters.dt, 
            perpetual_mark_price_ts=perpetual_mark_price_ts, 
            atm_factor_ts=atm_factor_ts
        )
        trader = VolatilityTrader(traderparam)
        return trader.update_book()
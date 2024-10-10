from dataclasses import dataclass
import operator
from datetime import datetime
from typing import List
import numpy as np
from src.instruments import (
    Currency, 
    Instrument, 
    Spot, 
    Option, 
    PerpetualFuture, 
    Future, 
    Sensitivities
)
from src.market import Market
from src.quant.blackscholes import BlackScholes

@dataclass
class CashFlow: 
    amount: float 
    currency : Currency

@dataclass
class Margin: 
    initial : float 
    maintenance : float 
    currency : Currency

@dataclass
class Trade:
    instrument : Instrument 
    number_contracts : float
    traded_price : float
    currency : Currency
    reference_time : datetime

    def __post_init__(self): 
        self.premium = self._set_premium() 

    def _set_premium(self) -> float: 
        if isinstance(self.instrument, Future): 
            return 0 
        elif isinstance(self.instrument, PerpetualFuture): 
            return 0 
        else: 
            return -self.number_contracts*self.traded_price

    def get_size(self) -> float: 
        n = self.number_contracts
        cz = self.instrument.contract_size
        if isinstance(self.instrument,Spot): return abs(n)
        if isinstance(self.instrument,Option): return abs(n)
        if isinstance(self.instrument,Future): 
            return abs(n*cz/self.traded_price)
        if isinstance(self.instrument,PerpetualFuture): 
            return abs(n*cz/self.traded_price)

    def trade_fee(self) -> CashFlow: 
        ccy = self.instrument.risk_factor.base_currency
        if isinstance(self.instrument,Spot): 
            return CashFlow(0,ccy)
        if isinstance(self.instrument,Option): 
            cap = 0.125*abs(self.number_contracts)*self.traded_price
            amount = -min(0.03*self.get_size()/100, cap)
            return CashFlow(amount,ccy)
        if isinstance(self.instrument,Future): 
            return CashFlow(-0.05*self.get_size()/100,ccy)
        if isinstance(self.instrument,PerpetualFuture):  
            return CashFlow(-0.05*self.get_size()/100,ccy)

@dataclass
class Position: 
    trades : List[Trade]

    def __post_init__(self): 
        self.number_contracts = self.get_net_position()
        self.instrument = self.trades[0].instrument
        self.price_currency = self.trades[0].currency
        self.fifo_price = self.get_fifo_price()
        self.premium = self.get_premium()

    def get_premium(self) -> float: 
        if isinstance(self.instrument, Future): 
            return 0 
        elif isinstance(self.instrument, PerpetualFuture): 
            return 0 
        else: 
            return self.number_contracts*self.fifo_price

    def get_net_position(self) -> float: 
        return round(sum([t.number_contracts for t in self.trades]),5)
    
    def get_fifo_price(self) -> float: 
        total_price = sum([t.number_contracts*t.traded_price 
                           for t in self.trades])
        try: return total_price/self.number_contracts
        except ZeroDivisionError: return 0
    
    def get_premium_cash_flow(self) -> CashFlow: 
        return CashFlow(
            -self.get_premium(), 
            self.price_currency)
    
    def get_fee_cash_flow(self) -> CashFlow: 
        amount = sum([t.trade_fee().amount for t in self.trades]) 
        ccy = [t.trade_fee().currency.code for t in self.trades][0]
        return CashFlow(amount, Currency(ccy))
    
    def get_realised_pnl_cash_flow(self) -> CashFlow: 
        total_price = sum([-t.number_contracts*t.traded_price 
                           for t in self.trades])
        total_price = round(total_price + self.number_contracts*self.fifo_price,5)
        return CashFlow(total_price,self.price_currency)  
    
    def to_trade(self, reference_time:datetime) -> Trade: 
        return Trade(
            instrument = self.instrument, 
            number_contracts=self.number_contracts, 
            traded_price=self.fifo_price,
            currency=self.price_currency, 
            reference_time=reference_time)

    def get_settlement_trade(
            self, 
            settlement_price: float, 
            reference_time:datetime) -> Trade:
        return Trade(
            instrument = self.instrument, 
            number_contracts=-self.number_contracts, 
            traded_price=settlement_price,
            currency=self.price_currency, 
            reference_time=reference_time) 
    
    def get_cash_flows(self) -> List[CashFlow]: 
        return [
            self.get_fee_cash_flow(), 
            self.get_realised_pnl_cash_flow(), 
            self.get_premium_cash_flow()
        ]

def trades_to_positions(trades : List[Trade]) -> List[Position]: 
    instrument_names = list(set([t.instrument.name for t in trades]))
    output = list()
    for n in instrument_names: 
        trade = [t for t in trades if t.instrument.name==n]
        output.append(Position(trade))
    return output

@dataclass
class Portfolio: 
    positions : List[Position] 
    market : Market 

    def __post_init__(self): 
        self.spot_quote = self.market.get_quote(self.market.spot.name)
        #self.positions = self.settle_expired_positions()
        self.sensitivities = self.get_usd_sensitivities('mid')

    def get_usd_sensitivities(self, quote_type:str) -> Sensitivities: 
        delta_exposure,gamma_exposure = list(), list()
        vega_exposure, theta_exposure = list(), list()
        futts = self.market.get_future_term_structure()
        for p in self.positions: 
            if p.number_contracts==0: continue
            q = p.number_contracts*p.instrument.contract_size
            if Option.__instancecheck__(p.instrument): 
                strike = p.instrument.strike
                t = p.instrument.time_to_expiry(self.market.reference_time)
                futprice = futts.future_price(t)
                sigma = self.market.get_implied_volatility_quote(
                    p.instrument.name,quote_type)
                if p.instrument.call_or_put == 'C': put_or_call = 1
                else:  put_or_call = -1
                bs = BlackScholes(
                    S = futprice, K = strike, t = t, sigma = sigma, 
                    q = 0, r=0, future=0, call_or_put=put_or_call)
                delta_exposure.append(q*bs.delta())
                gamma_exposure.append(q*bs.gamma())
                vega_exposure.append(q*bs.vega())
                theta_exposure.append(q*bs.theta())
            elif Spot.__instancecheck__(p.instrument): 
                delta_exposure.append(q)
                gamma_exposure.append(0)
                vega_exposure.append(0) 
                theta_exposure.append(0)
            else: 
                d =  q/self.spot_quote.order_book.mid
                delta_exposure.append(d)
                gamma_exposure.append(0)
                vega_exposure.append(0) 
                theta_exposure.append(0)
        return Sensitivities(
            delta = sum(delta_exposure), 
            gamma = sum(gamma_exposure), 
            theta = sum(theta_exposure), 
            vega = sum(vega_exposure))
    
    def get_cash_account(self, deposit:CashFlow) -> List[CashFlow]: 
        output_dict = {deposit.currency.code : deposit.amount}
        for p in self.positions: 
            cash_flows = p.get_cash_flows()
            for cf in cash_flows: 
                ccy, amount = cf.currency.code, cf.amount
                if ccy in output_dict:
                    output_dict[ccy] = output_dict[ccy]+amount
                else: output_dict[ccy] = amount
            if isinstance(p.instrument, Spot): 
                ccyspot = p.instrument.risk_factor.base_currency.code
                if ccyspot in output_dict:
                    output_dict[ccyspot] = output_dict[ccyspot]+p.number_contracts
                else: output_dict[ccyspot] = p.number_contracts
        return [CashFlow(round(output_dict[o],5),Currency(o)) for o in output_dict]

    def perpetual_delta_hedging_trade(self) -> Trade: 
        perp = self.market.perpetual
        quote = self.market.get_quote(perp.name)
        exposure = self.sensitivities.delta
        if exposure<=0: price = quote.order_book.best_ask
        else: price = quote.order_book.best_bid
        quantity = round(exposure*price/perp.contract_size)
        return Trade(
            instrument = perp,
            number_contracts=-quantity,
            traded_price=price,
            currency=quote.order_book.quote_currency, 
            reference_time=self.market.reference_time
        ) 

    def get_margin(self) -> Margin: 
        im, mm = list(), list()
        base_ccy = self.market.risk_factor.base_currency
        futts = self.market.get_future_term_structure()
        for p in self.positions:
            if p.number_contracts==0: continue
            instrument = p.instrument        
            if isinstance(instrument,Spot): mm_rate,im_rate,size = 0,0,0
            if isinstance(instrument,Option): 
                size = p.number_contracts*instrument.contract_size
                if size>=0: im_rate, mm_rate = 0,0
                else: 
                    quote = self.market.get_quote(instrument.name)
                    t = instrument.time_to_expiry(self.market.reference_time)
                    ump = futts.future_price(t)
                    mp = quote.order_book.mark_price
                    K = instrument.strike
                    if instrument.call_or_put == 'C': 
                        otm_amount = max(ump-K, 0) 
                        mm_rate = 0.075 + mp
                        im_rate = max(0.15-otm_amount/ump,0.1)+mp
                    else: 
                        otm_amount = max(K-ump, 0) 
                        mm_rate = max(0.075,0.075*mp)+mp
                        im_rate = max(max(0.15-otm_amount/ump,0.1)+mp,mm_rate)
            if isinstance(instrument,Future): 
                spot_price = self.spot_quote.order_book.mid
                size = p.number_contracts*instrument.contract_size/spot_price
                if base_ccy == Currency('ETH'):
                    im_rate = 0.04 + abs(size)*0.004/100
                    mm_rate = 0.02 + abs(size)*0.004/100
                if base_ccy == Currency('BTC'):
                    im_rate = 0.04 + abs(size)*0.005/100
                    mm_rate = 0.02 + abs(size)*0.005/100
            if isinstance(instrument,PerpetualFuture): 
                spot_price = self.spot_quote.order_book.mid
                size = p.number_contracts*instrument.contract_size/spot_price
                if base_ccy == Currency('ETH'):
                    im_rate = 0.02 + abs(size)*0.004/100
                    mm_rate = 0.01 + abs(size)*0.004/100
                if base_ccy == Currency('BTC'):
                    im_rate = 0.02 + abs(size)*0.005/100
                    mm_rate = 0.01 + abs(size)*0.005/100
            im.append(im_rate*abs(size))
            mm.append(mm_rate*abs(size))
        return Margin(sum(im),sum(mm),self.market.risk_factor.base_currency)
        
    def settle_expired_positions(self) -> List[Position]: 
        new_position = list()
        for p in self.positions: 
            if p.number_contracts == 0: 
                new_position.append(p)
                continue
            i = p.instrument
            S = self.spot_quote.order_book.mark_price 
            if isinstance(i,Future): 
                exp, ref = i.expiry_dt, self.market.reference_time
                if p.number_contracts!=0 and exp==ref: 
                    settle_trade = p.get_settlement_trade(S, ref)
                    new_trades = p.trades.copy()
                    new_trades.append(settle_trade)
                    new_position.append(Position(new_trades))
                else: new_position.append(p)
            elif isinstance(i,Option): 
                exp, ref = i.expiry_dt, self.market.reference_time
                if p.number_contracts!=0 and exp==ref: 
                    K = i.strike
                    if i.call_or_put=='C': x = 1 
                    else: x=-1
                    price = x*max(S-K, 0)/S
                    settle_trade = p.get_settlement_trade(price, ref)
                    new_trades = p.trades.copy()
                    new_trades.append(settle_trade)
                    new_position.append(Position(new_trades))
                else: new_position.append(p)
            else: new_position.append(p) 
        return new_position

    def is_cash_sufficient(self, deposit:CashFlow) -> bool: 
        margin = self.get_margin()
        try:
            cash_account = [ca for ca in self.get_cash_account(deposit) 
                            if ca.currency==margin.currency][0]
        except IndexError: 
            cash_account = CashFlow(0,margin.currency)
        if cash_account.amount - margin.initial > margin.maintenance: 
            return True
        else: return False

    def get_position_usd_unrealised_pnl(self, position:Position) -> float: 
        spot_price = self.spot_quote.order_book.mid
        i = position.instrument
        if isinstance(i, PerpetualFuture):
            n = position.number_contracts*i.contract_size/spot_price
        else:
            n = position.number_contracts*i.contract_size
        fifo_price = position.fifo_price
        quote = self.market.get_quote(i.name)
        quoted_price = quote.order_book.mark_price
        if quote.order_book.quote_currency != Currency('USD'):
            return n*(quoted_price-fifo_price)*spot_price
        else: return n*(quoted_price-fifo_price)

    def get_usd_unrealised_pnl(self) -> float: 
        unrealised_pnl = list()
        for p in self.positions: 
            if p.number_contracts == 0: unrealised_pnl.append(0)
            else: unrealised_pnl.append(self.get_position_usd_unrealised_pnl(p))
        return sum(unrealised_pnl)
    
    def get_usd_realised_pnl(self, deposit: CashFlow) -> float: 
        cash_accounts = self.get_cash_account(deposit)
        cash_accounts_usd_value = 0 
        spot_price = self.spot_quote.order_book.mid
        for ca in cash_accounts: 
            if ca.currency == Currency('USD'): 
                cash_accounts_usd_value = cash_accounts_usd_value + ca.amount
            else: 
                value = ca.amount*spot_price
                cash_accounts_usd_value = cash_accounts_usd_value + value
        return cash_accounts_usd_value

    def get_usd_value(self, deposit: CashFlow) -> float: 
        return self.get_usd_realised_pnl(deposit)+self.get_usd_unrealised_pnl()

    def get_usd_fee_value(self) -> float: 
        output = 0 
        spot_price = self.spot_quote.order_book.mid
        for p in self.positions: 
            feecf = p.get_fee_cash_flow()
            if feecf.currency!=Currency('USD'): 
                output = output + spot_price*feecf.amount
            else: output = output + feecf.amount
        return output 





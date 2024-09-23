from dataclasses import dataclass
from datetime import datetime
from typing import List
import numpy as np
from src.market import (
    Currency, 
    Instrument, 
    Spot, 
    Option, 
    PerpetualFuture, 
    Future
)

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
    marked_price : float 
    currency : Currency
    trade_date_time : datetime

    def crypto_size(self) -> float: 
        n = self.number_contracts
        cz = self.instrument.contract_size
        if isinstance(self.instrument,Spot): return n
        if isinstance(self.instrument,Option): return n
        if isinstance(self.instrument,Future): 
            return n*cz/self.marked_price
        if isinstance(self.instrument,PerpetualFuture): 
            return n*cz/self.marked_price

    def trade_fee(self) -> CashFlow: 
        ccy = self.instrument.risk_factor.base_currency
        if isinstance(self.instrument,Spot): 
            return CashFlow(0,ccy)
        if isinstance(self.instrument,Option): 
            cap = 0.125*self.number_contracts*self.marked_price
            amount = -min(0.03*self.crypto_size/100, cap)
            return CashFlow(amount,ccy)
        if isinstance(self.instrument,Future): 
            return CashFlow(-0.05*self.crypto_size/100,ccy)
        if isinstance(self.instrument,PerpetualFuture):  
            return CashFlow(-0.05*self.crypto_size/100,ccy)
    
    def pay_leg_cash_flow(self) -> CashFlow: 
        if isinstance(self.instrument,Spot): 
            ccy = self.instrument.risk_factor.quote_currency
            amount = -self.number_contracts*self.traded_price
        if isinstance(self.instrument,Option): 
            ccy = self.instrument.risk_factor.base_currency
            amount = -self.number_contracts*self.traded_price
            return CashFlow(amount, ccy)
        if isinstance(self.instrument,Future): 
            ccy = self.instrument.risk_factor.quote_currency
            amount = 0
        if isinstance(self.instrument,PerpetualFuture): 
            ccy = self.instrument.risk_factor.quote_currency 
            amount = 0
        return CashFlow(amount, ccy)
    
    def receive_leg_cash_flow(self) -> CashFlow: 
        ccy = self.instrument.risk_factor.base_currency
        if isinstance(self.instrument,Spot): 
            amount = self.number_contracts
        else: amount = 0
        return CashFlow(amount, ccy)
    
    def settlement_cash_flow(self, settle_underlying_price:float) -> CashFlow: 
        n = self.number_contracts
        S = settle_underlying_price
        ccy = self.instrument.risk_factor.base_currency 
        if isinstance(self.instrument,Spot): amount = 0
        if isinstance(self.instrument,PerpetualFuture): amount = 0
        if isinstance(self.instrument,Future): 
            cz = self.instrument.contract_size
            amount =  n*cz*(S-self.traded_price)/S
        if isinstance(self.instrument,Option): 
            K = self.instrument.strike
            if self.Instrument.put_or_call =='C': p = 1
            else: p= -1
            amount = n*max(p*(S-self.instrument.strike),0)
        return CashFlow(amount, ccy)

    def initial_cash_flow(self) -> List[CashFlow]: 
        fee = self.trade_fee()
        pay = self.pay_leg_cash_flow()
        receive = self.receive_leg_cash_flow()
        dict_out = {}
        for cf in [fee, pay, receive]: 
            if cf.currency in list(dict_out.keys()):
                dict_out[cf.currency] = dict_out[cf.currency] + cf.amount
            else: dict_out[cf.currency] = cf.amount
        output = list()
        for k in list(dict_out.keys()): 
            output.append(CashFlow(dict_out[k], k))
        return output
    
    def margin(self, mark_price:float, underlying_mark_price:float) -> Margin: 
        mp, ump = mark_price, underlying_mark_price
        ccy = self.instrument.risk_factor.base_currency
        crypto_size = self.crypto_size()
        if isinstance(self.instrument,Spot): mm_rate,im_rate = 0,0
        if isinstance(self.instrument,Option): 
            if crypto_size>=0: im, mm = 0,0
            else: 
                K = self.instrument.strike
                if self.call_or_put == 'C': 
                    otm_amount = max(ump-K, 0) 
                    mm_rate = 0.075 + mp
                    im_rate = max(0.15-otm_amount/ump,0.1)+mp
                else: 
                    otm_amount = max(K-ump, 0) 
                    mm_rate = max(0.075,0.075*mp)+mp
                    im_rate = max(max(0.15-otm_amount/ump,0.1)+mp,mm_rate)
        if isinstance(self.instrument,Future): 
            if self.risk_factor.base_currency == Currency('ETH'):
                im_rate = 0.04 + abs(crypto_size)*0.004/100
                mm_rate = 0.02 + abs(crypto_size)*0.004/100
            if self.risk_factor.base_currency == Currency('BTC'):
                im_rate = 0.04 + abs(crypto_size)*0.005/100
                mm_rate = 0.02 + abs(crypto_size)*0.005/100
        if isinstance(self.instrument,PerpetualFuture): 
            if self.risk_factor.base_currency == Currency('ETH'):
                im_rate = 0.02 + abs(crypto_size)*0.004/100
                mm_rate = 0.01 + abs(crypto_size)*0.004/100
            if self.risk_factor.base_currency == Currency('BTC'):
                im_rate = 0.02 + abs(crypto_size)*0.005/100
                mm_rate = 0.01 + abs(crypto_size)*0.005/100
        im = im_rate*abs(crypto_size)
        mm = mm_rate*abs(crypto_size)
        return Margin(im,mm,ccy)


@dataclass
class Portfolio: 
    trades : List[Trade] 


@dataclass
class GammaTrader: 
    pass 



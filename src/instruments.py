from dataclasses import dataclass
from datetime import datetime 
from typing import List
from abc import ABC

mapping_deribit_month_number = {
    'JAN' : 1, 
    'FEB' : 2, 
    'MAR' : 3, 
    'APR' : 4, 
    'MAY' : 5, 
    'JUN' : 6, 
    'JUL' : 7, 
    'AUG' : 8, 
    'SEP' : 9, 
    'OCT' : 10, 
    'NOV' : 11, 
    'DEC' : 12
}

@dataclass
class Currency:
    code : str 

currency = {
    'USD' : Currency('USD'),
    'BTC' : Currency('BTC'), 
    'ETH' : Currency('ETH')
}

@dataclass
class RiskFactor:
    base_currency : Currency
    quote_currency : Currency
    
    def __post_init__(self): 
        self.code = self.base_currency.code+self.quote_currency.quote

def deribit_date_to_datetime(deritbit_date: str) -> datetime: 
    hour = 8 
    if len(deritbit_date)==6: 
        day = int(deritbit_date[0:1])
        year = int('20'+deritbit_date[4:6])
        month = mapping_deribit_month_number[deritbit_date[1:4]]
    else: 
        day = int(deritbit_date[0:2])
        year = int('20'+deritbit_date[5:7])
        month = mapping_deribit_month_number[deritbit_date[2:5]]
    return datetime(year,month,day,hour)

@dataclass 
class OrderBook: 
    quote_currency : Currency
    mark_price : float
    best_bid : float 
    best_ask : float 
    best_bid_size : float 
    best_ask_size : float 
    asks : List[float]
    bids : List[float]
    volume_usd : float

@dataclass
class OptionData: 
    bid_iv : float 
    ask_iv : float 
    delta : float 
    gamma : float 
    theta : float 
    vega : float 

@dataclass
class TradingFee: 
    currency : Currency
    value : float 

@dataclass
class Margin: 
    currency : Currency
    maintenance : float 
    initial : float

@dataclass
class Instrument(ABC): 
    reference_time : datetime 
    name : str 
    order_book : OrderBook
    volume_usd : float
    open_interest : float 
    risk_factor : RiskFactor

    def __post_init__(self): 
        self.split_name = self.name.split('-')
        self.risk_factor = RiskFactor(
            base_currency = self.split_name[0], 
            quote_currency = currency['USD'])
    
    @abstractmethod
    def get_margin(self, size:float) -> Margin: 
        pass 
    
    @abstractmethod
    def get_trading_fees(self) -> TradingFee: 
        pass 

@dataclass 
class PerpetualFuture(Instrument): 
    
    def get_margin(self, size:float) -> Margin: 
        if self.risk_factor.base_currency == currency['ETH']:
            im_rate = 0.02 + abs(size)*0.004/100
            mm_rate = 0.01 + abs(size)*0.004/100
        if self.risk_factor.base_currency == currency['BTC']:
            im_rate = 0.02 + abs(size)*0.005/100
            mm_rate = 0.01 + abs(size)*0.005/100
        return Margin(
            currency=self.order_book.quote_currency, 
            maintenance = abs(size)*mm_rate, 
            initial = abs(size)*im_rate)
    
    def get_trading_fees(self) -> TradingFee: 
        fee = 0.05*self.order_book.mark_price/100 
        return TradingFee(self.order_book.quote_currency, fee)   

@dataclass
class Future(Instrument): 

    def __post_init__(self):
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 

    def get_margin(self, size:float) -> Margin: 
        if self.risk_factor.base_currency == Currency('ETH'):
            im_rate = 0.04 + abs(size)*0.004/100
            mm_rate = 0.02 + abs(size)*0.004/100
        if self.risk_factor.base_currency == Currency('BTC'):
            im_rate = 0.04 + abs(size)*0.005/100
            mm_rate = 0.02 + abs(size)*0.005/100
        return Margin(
            currency=self.order_book.quote_currency, 
            maintenance = abs(size)*mm_rate, 
            initial = abs(size)*im_rate)
    
    def get_trading_fees(self) -> TradingFee: 
        fee = 0.05*self.order_book.mark_price/100
        return TradingFee(self.order_book.quote_currency, fee)  
    
@dataclass
class Option(Instrument): 
    data : OptionData
    underlying : Instrument

    def __post_init__(self):
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 
        self.strike = int(self.split_name[2])
        self.call_or_put = self.split_name[3]
        
    def get_otm_amount(self) -> float: 
        S, K = self.strike, self.underlying.order_book.mark_price
        if self.call_or_put == 'C': 
            return max(K-S, 0) 
        else: 
            return max(S-K, 0)   
    
    def get_margin(self, size:float) -> Margin: 
        ccy = self.order_book.quote_currency
        mp = self.order_book.mark_price
        ump = self.underlying.order_book.mark_price
        if size>0: 
            im, mm = 0,0
        else: 
            if self.call_or_put == 'C': 
                mm = 0.075 + mp
                im = max(0.15-self.get_otm_amount()/ump,0.1)+mp
            else: 
                mm = max(0.075,0.075*mp)+mp
                im = max(max(0.15-self.get_otm_amount()/ump,0.1)+mp,mm)
        return Margin(ccy, mm, im)
    
    def get_trading_fees(self) -> TradingFee: 
        flat = 0.03*self.underlying.order_book.mark_price/100
        cap = 0.125*self.order_book.mark_price
        return TradingFee(self.order_book.quote_currency, min(flat, cap))
    



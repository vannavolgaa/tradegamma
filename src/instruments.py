from dataclasses import dataclass
from datetime import datetime 
from typing import List
from abc import ABC, abstractmethod

def deribit_date_to_datetime(deritbit_date: str) -> datetime: 
    hour = 8 
    mapping_deribit_month_number = {
    'JAN' : 1, 'FEB' : 2, 'MAR' : 3, 'APR' : 4, 
    'MAY' : 5, 'JUN' : 6, 'JUL' : 7, 'AUG' : 8, 
    'SEP' : 9, 'OCT' : 10, 'NOV' : 11, 'DEC' : 12}
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
class Currency:
    code : str 

@dataclass
class RiskFactor:
    base_currency : Currency
    quote_currency : Currency
    
    def __post_init__(self): 
        self.code = self.base_currency.code+self.quote_currency.code

@dataclass
class Instrument(ABC): 
    name : str 
    risk_factor : RiskFactor
    underlying_name : str

    def __post_init__(self): 
        self.split_name = self.name.split('-')

@dataclass 
class PerpetualFuture(Instrument): 
    pass 

@dataclass
class Future(Instrument): 

    def __post_init__(self):
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 

@dataclass
class Spot(Instrument): 
    pass 
    
@dataclass
class Option(Instrument): 

    def __post_init__(self):
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 
        self.strike = int(self.split_name[2])
        self.call_or_put = self.split_name[3]

@dataclass 
class OrderBook:
    reference_time : datetime
    quote_currency : Currency
    mark_price : float
    best_bid : float 
    best_ask : float 
    best_bid_size : float 
    best_ask_size : float 
    asks : List[float]
    bids : List[float]

@dataclass
class Sensitivities: 
    delta : float 
    gamma : float 
    theta : float 
    vega : float 

@dataclass
class InstrumentQuote: 
    instrument_name : str 
    order_book : OrderBook
    volume_usd : float
    open_interest : float 
    sensitivities : Sensitivities
    bid_iv : float 
    ask_iv : float 


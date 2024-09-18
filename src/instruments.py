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
class Instrument(ABC): 
    reference_time : datetime 
    name : str 
    order_book : OrderBook
    volume_usd : float
    open_interest : float 

    def __post_init__(self): 
        self.split_name = self.name.split('-')
        self.asset = self.split_name[0]

@dataclass 
class PerpetualFuture(Instrument): 
    pass

@dataclass
class Future(Instrument): 

    def __post_init__(self):
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 

@dataclass
class Option(Instrument): 
    data : OptionData
    underlying : Instrument

    def __post_init__(self):
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 
        self.strike = int(self.split_name[2])
        self.call_or_put = self.split_name[3]


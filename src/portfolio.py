from dataclasses import dataclass
from datetime import datetime 
from typing import List 
from instruments2 import Instrument, Currency, Margin


@dataclass
class Trade:
    traded_time : datetime 
    instrument : Instrument 
    quantity : float 
    traded_price : float
    currency : Currency



@dataclass
class Portfolio: 
    trades : List[Trade]
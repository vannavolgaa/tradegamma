from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
import operator 

@dataclass
class Trade: 
    ref_time : datetime 
    quantity : float 
    price : float 


@dataclass
class Position: 
    trades : List[Trade]
    
    def __post_init__(self): 
        self.net_position = self.get_net_position()
        self.fifo_price = self.get_fifo_price()
        self.premium = self.get_premium()
    
    def get_net_position(self) -> float: 
        return sum([t.quantity for t in self.trades])
    
    def get_fifo_price(self) -> float: 
        np = self.net_position
        trades = sorted(self.trades, key=operator.attrgetter('ref_time'),reverse=True)
        xmax, xmin = self.net_position,self.net_position
        cfmax, cfmin = list(), list()
        for t in trades: 
            qmax, qmin = max(xmax-max(t.quantity,0), 0), min(xmin-min(t.quantity,0), 0)
            deltamax, deltamin = xmax-qmax, xmin-qmin
            cfmax.append(deltamax*t.price), cfmin.append(deltamin*t.price)
            xmax, xmin = qmax, qmin
        if self.net_position == 0: return 0 
        elif self.net_position>0: return sum(cfmax)/self.net_position
        elif self.net_position<0: return sum(cfmin)/self.net_position
    
    def get_premium(self) -> float: 
        return -self.net_position*self.fifo_price
    
    def get_realised_pnl(self) -> float: 
        total_cash_flow = sum([-t.quantity*t.price for t in self.trades])
        premium = -self.net_position*self.fifo_price
        return total_cash_flow-premium

dates = [datetime.now() + timedelta(days=i) for i in range(0,22)]
contracts = [-7,
8,
-8,
4,
-3,
-7,
-9,
-5,
7,
4,
-8,
10,
5,
-3,
-4,
-4,
-1,
-7,
3,
7,
-2,
1]


prices = [31,
33,
30,
35,
25,
32,
28,
35,
28,
27,
27,
29,
33,
31,
33,
34,
26,
26,
33,
28,
26,
27]

trade = [Trade(d,c,p) for d,c,p in zip(dates,contracts,prices)]
position = Position(trade)

print(position.get_net_position())
print(position.get_fifo_price())
print(position.get_premium())
print(position.get_realised_pnl())

from src.dataloader import load_data
from dataclasses import dataclass
from datetime import datetime 
from typing import List
from abc import ABC

n = 500000
data = load_data(n)

ts = [d['timestamp_call']for d in data]

ts0 = ts[0]
tslast = ts[n-1]


# what drives sqrt((dS/S)**2 * 1/dt) - sigma du perpetual
# what drives the sigma_dt - sigma  
# trade it 
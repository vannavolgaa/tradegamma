import pandas as pd 
from typing import List

def load_data(n:int = 100) -> List[dict]: 
    data = pd.read_csv('data/aggregate_deribit_data.csv', nrows=n)
    return data.to_dict('records')
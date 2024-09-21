from typing import List
import pandas as pd 
import math 
from datetime import datetime
from src.instruments import (
    Instrument, 
    InstrumentQuote, 
    Option, 
    Spot, 
    Future, 
    PerpetualFuture, 
    OrderBook, 
    Currency,
    RiskFactor, 
    Sensitivities
)

class MarketLoader: 
    def __init__(self, n:int) -> None:
        self.n = n 
        self.data = self._get_data()
        self.dates = [d['timestamp_call']for d in self.data]
        self.dates_dt = [self._process_date_to_datetime(d) for d in self.dates]
        self.usd = Currency('USD')
        self.btc = Currency('BTC')
        self.eth = Currency('ETH')
        self.btcusd = RiskFactor(self.btc, self.usd)
        self.ethusd = RiskFactor(self.eth, self.usd)
        self.quotes = self._process_quotes()
        self.instruments = self._process_instruments()
    
    @staticmethod
    def _process_date_to_datetime(date:str) -> datetime: 
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def _filter_dates(data: List[dict]) -> List[str]:
        n = len(data)
        ts = [d['timestamp_call']for d in data]
        ts0 = ts[0]
        tslast = ts[n-1] 
        tslist = [ts0, tslast]
        return [t for t in ts if t not in tslist]
    
    def _get_data(self) -> List[dict]: 
        data = pd.read_csv('data/aggregate_deribit_data.csv', nrows=self.n)
        data = data.to_dict('records')
        dates = self._filter_dates(data)
        return [d for d in data if d['timestamp_call'] in dates]
    
    @staticmethod
    def _process_instrument_type(data:dict) -> str: 
        name = data['instrument_name']
        if 'C' in name: return 'option'
        elif 'P' in name: return 'option'
        elif 'PERPETUAL' in name: return 'perpetual'
        else: return 'future'
    
    def _process_risk_factor(self, data:dict) -> RiskFactor: 
        if 'ETH' in data['instrument_name'] : return self.ethusd
        if 'BTC' in data['instrument_name'] : return self.btcusd
    
    def _get_spot_instruments(self) -> List[Spot]: 
        btc_name = 'SPOT'+self.btcusd.code 
        eth_name = 'SPOT'+self.ethusd.code 
        return [Spot(btc_name, self.btcusd, None), 
                Spot(eth_name, self.ethusd, None)]
    
    def _process_spot_quotes(self) -> List[InstrumentQuote]: 
        output = list()
        empty_list = list()
        for dt in self.dates: 
            dataset = [d for d in self.data if d['timestamp_call']==dt]
            eth = [d for d in dataset if 'ETH' in d['instrument_name']][0]
            eth_price = eth['index_price']
            btc = [d for d in dataset if 'BTC' in d['instrument_name']][0]
            btc_price = btc['index_price']
            eth_quote = InstrumentQuote(
                instrument_name='SPOT'+self.ethusd.code, 
                order_book=OrderBook(dt, self.usd, eth_price,
                                     eth_price,eth_price,math.nan,
                                     math.nan,empty_list,empty_list,
                                     math.nan), 
                volume_usd=math.nan, 
                sensitivities=Sensitivities(1,0,0,0),
                bid_iv=math.nan, 
                ask_iv=math.nan)
            btc_quote = InstrumentQuote(
                instrument_name='SPOT'+self.btcusd.code, 
                order_book=OrderBook(dt, self.usd, btc_price,
                                     btc_price,btc_price,math.nan,
                                     math.nan,empty_list,empty_list,
                                     math.nan), 
                volume_usd=math.nan, 
                sensitivities=Sensitivities(1,0,0,0),
                bid_iv=math.nan, 
                ask_iv=math.nan)
            output.append(btc_quote)
            output.append(eth_quote)
        return output

    def _process_quote_currency(self, data:dict) -> Currency: 
        itype = self._process_instrument_type(data)
        match itype: 
            case 'option': 
                if 'ETH' in data['instrument_name']: return self.eth
                if 'BTC' in data['instrument_name']: return self.btc
            case 'future': return self.usd
            case 'perpetual': return self.usd

    def _process_order_book(self, data:dict) -> OrderBook:
        return OrderBook(
            reference_time=self._process_date_to_datetime(data['timestamp_call']),
            quote_currency=self._process_quote_currency(data),
            mark_price=data['mark_price'],
            best_ask=data['best_ask_price'],
            best_bid=data['best_bid_price'],
            best_ask_size=data['best_ask_amount'],
            best_bid_size=data['best_bid_amount'],
            asks=data['asks'][0], 
            bids=data['bids'][0]
        )
    
    def _process_sensitivities(self, data: dict) -> Sensitivities: 
        itype = self._process_instrument_type(data)
        match itype: 
            case 'option': 
                return Sensitivities(
                    delta = data['greeks.delta'], 
                    gamma = data['greeks.gamma'], 
                    theta = data['greeks.theta'], 
                    vega = data['greeks.vega'], 
                )
            case 'future': return Sensitivities((1,0,0,0))
            case 'perpetual': return Sensitivities((1,0,0,0))
    
    def _process_instrument(self, data:dict) -> Instrument: 
        itype = self._process_instrument_type(data)
        match itype: 
            case 'option': 
                return Option(
                    name = data['instrument_name'],
                    risk_factor=self._process_risk_factor(data), 
                    underlying_name=data['instrument_name'])
            case 'future': 
                return Future(
                    name = data['instrument_name'],
                    risk_factor=self._process_risk_factor(data), 
                    underlying_name=data['instrument_name'])
            case 'perpetual': 
                return PerpetualFuture(
                    name = data['instrument_name'],
                    risk_factor=self._process_risk_factor(data), 
                    underlying_name=data['instrument_name'])
    
    def _process_instruments(self) -> List[Instrument]: 
        names = [d['instrument_name'] for d in self.data]
        unames = list(set(names))
        output = list()
        for n in unames: 
            dataset = [d for d in self.data if d['instrument_name'][0] == n][0]
            output.append(self._process_instrument(dataset))
        return output + self._get_spot_instruments()

    def _process_quote(self, data:dict) -> InstrumentQuote: 
        return InstrumentQuote(
            instrument_name = data['instrument_name'], 
            order_book=self._process_order_book(data), 
            sensitivities=self._process_sensitivities(data),
            bid_iv = data['bid_iv'], 
            ask_iv = data['ask_iv'], 
            volume_usd = data['stats.volume_usd'], 
            open_interest = data['open_interest'], 
        ) 

    def _process_quotes(self) -> List[InstrumentQuote]: 
        instruments = [self._process_quote(d) for d in self.data]
        return instruments + self._process_spot_quotes() 


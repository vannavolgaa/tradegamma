from dataclasses import dataclass
from typing import List
import pandas as pd 
import math 
from datetime import datetime, timedelta
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
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
from src.quant.timeserie import TimeSerie
from src.tools import update_dict_of_list
from src.market import Market

garch_time_delta: timedelta = timedelta(days = 10)

auto_regressive_time_delta: timedelta = timedelta(days = 10)

def get_deribit_data() -> pd.DataFrame: 
    with open('data/deribit_data.pkl', 'rb') as inp:
        return pickle.load(inp)

def update_deribit_data() -> None: 
    data = pd.read_csv('data/aggregate_deribit_data.csv')
    with open('data/deribit_data.pkl', 'wb') as outp:  
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

class MarketLoader: 
    def __init__(self, nstart:int=0, nend:int=1000000) -> None:
        self.nstart, self.nend = nstart, nend
        self._set_data()
        self.quotes = self._process_quotes()
        self.instruments = self._process_instruments()
        self._set_advanced_data()
        self.markets = self._process_markets()
    
    @staticmethod
    def _process_date_to_datetime(date:str) -> datetime: 
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    
    def _set_dates(self, data: List[dict]) -> None:
        dates = list(set([d['timestamp_call']for d in data]))
        dates_dt = [self._process_date_to_datetime(d) for d in dates]
        ts0 = min(dates_dt)
        tslast = max(dates_dt)
        tslist = [ts0, tslast]
        self.dates_dt = [d for d in dates_dt if d not in tslist]
        self.dates_dt.sort()
        self.dates = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_dt]
    
    def _set_data(self) -> None: 
        data = get_deribit_data()
        data = data.iloc[self.nstart:self.nend]
        data = data.to_dict('records')
        self._set_dates(data)
        self.data, self.mapped_instrument_data = list(), dict()
        for d in data: 
            if d['timestamp_call'] in self.dates: 
                self.data.append(d)
                name = d['instrument_name']
                if name not in list(self.mapped_instrument_data.keys()):
                    self.mapped_instrument_data[name] = d
        self.usd = Currency('USD')
        self.btc = Currency('BTC')
        self.eth = Currency('ETH')
        self.btcusd = RiskFactor(self.btc, self.usd)
        self.ethusd = RiskFactor(self.eth, self.usd)
        print('Initial data loading is done.')
    
    def _set_advanced_data(self) -> None: 
        self.mapped_dates_quotes = dict()
        self.mapped_riskfactor_instruments = dict()
        self.mapped_dates_quotes_iname = dict()
        for q in self.quotes: 
            rtime = q.reference_time
            self.mapped_dates_quotes = update_dict_of_list(
                rtime,q,self.mapped_dates_quotes)
        for i in self.instruments: 
            rf = i.risk_factor.code
            self.mapped_riskfactor_instruments = update_dict_of_list(
                rf,i,self.mapped_riskfactor_instruments)
        for k in list(self.mapped_dates_quotes.keys()):
            names = [q.instrument_name 
                     for q in self.mapped_dates_quotes[k]]
            self.mapped_dates_quotes_iname[k] = names 

    @staticmethod
    def _process_instrument_type(data:dict) -> str: 
        name = data['instrument_name']
        split_name = name.split('-')
        if len(split_name) == 4: return 'option'
        else: 
            if 'PERPETUAL' in name: return 'perpetual'
            else: return 'future'
           
    def _process_risk_factor(self, data:dict) -> RiskFactor: 
        if 'ETH' in data['instrument_name'] : return self.ethusd
        if 'BTC' in data['instrument_name'] : return self.btcusd
    
    def _get_spot_instruments(self) -> List[Spot]: 
        btc_name = 'SPOT'+self.btcusd.code 
        eth_name = 'SPOT'+self.ethusd.code 
        return [Spot(btc_name, self.btcusd, None,1,self.btcusd.base_currency,0.0001), 
                Spot(eth_name, self.ethusd, None,1,self.btcusd.base_currency,0.0001)]
    
    def _process_spot_quotes(self) -> List[InstrumentQuote]: 
        output = list()
        empty_list = list()
        btc_perpetual = [d for d in self.data 
                         if d['instrument_name']=='BTC-PERPETUAL']
        eth_perpetual = [d for d in self.data 
                         if d['instrument_name']=='ETH-PERPETUAL']
        self.all_btc_dates = list()
        self.all_eth_dates = list()
        for b in btc_perpetual: 
            btc_price = b['index_price']
            dt = b['timestamp_call']
            btc_quote = InstrumentQuote(
                reference_time=self._process_date_to_datetime(dt),
                instrument_name='SPOT'+self.btcusd.code, 
                order_book=OrderBook(self.usd, btc_price,
                                     btc_price,btc_price,math.nan,
                                     math.nan,empty_list,empty_list), 
                volume_usd=math.nan, 
                sensitivities=Sensitivities(1,0,0,0),
                bid_iv=math.nan, 
                ask_iv=math.nan, 
                open_interest=math.nan)
            output.append(btc_quote)
        for e in eth_perpetual: 
            eth_price = e['index_price']
            dt = e['timestamp_call']
            eth_quote = InstrumentQuote(
                reference_time=self._process_date_to_datetime(dt),
                instrument_name='SPOT'+self.ethusd.code, 
                order_book=OrderBook(self.usd, eth_price,
                                     eth_price,eth_price,math.nan,
                                     math.nan,empty_list,empty_list), 
                volume_usd=math.nan, 
                sensitivities=Sensitivities(1,0,0,0),
                bid_iv=math.nan, 
                ask_iv=math.nan, 
                open_interest=math.nan)
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
            case 'future': return Sensitivities(1,0,0,0)
            case 'perpetual': return Sensitivities(1,0,0,0)
    
    def _process_instrument(self, data:dict) -> Instrument: 
        itype = self._process_instrument_type(data)
        rf = self._process_risk_factor(data)
        match itype: 
            case 'option': 
                return Option(
                    name = data['instrument_name'],
                    risk_factor=rf, 
                    underlying_name=data['instrument_name'], 
                    contract_size=1, 
                    contract_size_currency = rf.base_currency, 
                    minimum_contract_amount=0.1)
            case 'future': 
                return Future(
                    name = data['instrument_name'],
                    risk_factor=rf, 
                    underlying_name=data['instrument_name'], 
                    contract_size=10, 
                    contract_size_currency=self.usd, 
                    minimum_contract_amount=1)
            case 'perpetual': 
                return PerpetualFuture(
                    name = data['instrument_name'],
                    risk_factor=rf, 
                    underlying_name=data['instrument_name'], 
                    contract_size=10, 
                    contract_size_currency=self.usd, 
                    minimum_contract_amount=1)
    
    def _process_instruments(self) -> List[Instrument]: 
        names = list(self.mapped_instrument_data.keys())
        output = list()
        i = 0
        nn = len(names)
        for n in names: 
            i = i+1
            dataset = self.mapped_instrument_data[n]
            output.append(self._process_instrument(dataset))
            perc = round(100*i/nn,2)
            print(str(perc) + '% instruments processed')
        print('Instruments processing is done.')
        return output + self._get_spot_instruments()

    def _process_quote(self, data:dict) -> InstrumentQuote: 
        return InstrumentQuote(
            reference_time=self._process_date_to_datetime(data['timestamp_call']),
            instrument_name = data['instrument_name'], 
            order_book=self._process_order_book(data), 
            sensitivities=self._process_sensitivities(data),
            bid_iv = data['bid_iv'], 
            ask_iv = data['ask_iv'], 
            volume_usd = data['stats.volume_usd'], 
            open_interest = data['open_interest'], 
        ) 

    def _process_quotes(self) -> List[InstrumentQuote]: 
        n = len(self.data)
        output = list()
        i = 0
        for d in self.data: 
            i = i+1
            output.append(self._process_quote(d))
            perc = round(100*i/n,2)
            print(str(perc) + '% quote processed')
        print('Quotes processing is done.')
        return output + self._process_spot_quotes() 
    
    def _process_market(
            self, 
            reference_time: datetime, 
            riskf:RiskFactor) -> Market:
        instruments = self.mapped_riskfactor_instruments[riskf.code]
        quotes = self.mapped_dates_quotes[reference_time]
        qnames = self.mapped_dates_quotes_iname[reference_time]
        options, futures = list(), list()
        for i in instruments: 
            if i.name not in qnames: continue
            if Spot.__instancecheck__(i): spot = i
            if PerpetualFuture.__instancecheck__(i): perp = i
            if Future.__instancecheck__(i): futures.append(i)
            if Option.__instancecheck__(i): options.append(i)
        return Market(reference_time,riskf,quotes,spot,perp,futures,options)
    
    def _process_markets(self) -> List[Market]: 
        market_risk_factors = [self.btcusd]
        output = list()
        n = len(self.dates_dt)
        for i in range(0, len(self.dates_dt)): 
            d = self.dates_dt[i]
            d_1 = self.dates_dt[i-1]
            for m in market_risk_factors:
                try: 
                    output.append(self._process_market(d,m))
                except Exception as e: 
                    output.append(self._process_market(d_1,m))
            perc = round(100*i/n,2)
            print(str(perc) + '% market processed')
        print('Markets processing is done.')
        return output 

    def find_quote(self, name:str, reftime: datetime) -> InstrumentQuote: 
        return [q for q in self.quotes 
                if q.instrument_name==name and q.reference_time==reftime][0] 

    def find_instrument(self, name:str) -> Instrument: 
        return [q for q in self.instruments if q.name==name][0]
    
    def get_instrument_mark_price_time_serie(
            self, 
            instrument_name: str, 
            reference_time: datetime, 
            time_delta:timedelta) -> TimeSerie: 
        start_date = reference_time - time_delta
        quotes = [q for q in self.quotes if q.instrument_name==instrument_name]
        fquotes = [q for q in quotes 
                   if q.reference_time >= start_date 
                   and q.reference_time<=reference_time]
        markprices = [fq.order_book.mark_price for fq in fquotes]
        dates = [fq.reference_time for fq in fquotes]
        return TimeSerie(dict(zip(dates,markprices)))

    def get_risk_factor_atm_factor_time_serie(
            self, 
            risk_factor: RiskFactor, 
            reference_time: datetime, 
            time_delta:timedelta) -> TimeSerie: 
        start_date = reference_time - time_delta
        output = dict()
        for m in self.markets: 
            mdate = m.reference_time
            if mdate>=start_date and mdate<=reference_time: 
                if m.risk_factor == risk_factor: 
                    output[mdate] = m.atm_factor
        return TimeSerie(output)

    def get_instrument_log_return(
            self, 
            instrument_name: str, 
            reference_time: datetime, 
            time_delta:timedelta) -> float: 
        ts = self.get_instrument_mark_price_time_serie(
            instrument_name, reference_time, time_delta, True)
        dates = list(ts.datamap.keys())
        start, end = ts.datamap[min(dates)], ts.datamap[max(dates)]
        return np.log(end) - np.log(start)
    
    def plot_realized_volatility_model_fit(self, 
            instrument_name: str, 
            reference_time: datetime, 
            time_delta:timedelta) -> None: 
        ts = self.get_instrument_mark_price_time_serie(
            instrument_name, reference_time, time_delta)
        egarch = ts.normal_egarch_fit()
        vol = egarch.conditional_volatility
        ret = ts.log_difference
        dates = list(ts.datamap.keys())
        dates = dates[1:len(dates)]
        title = instrument_name + ' fit @ ' + \
        reference_time.strftime('%Y-%m-%d %H:%M:%S')
        plt.plot(dates,ret)
        plt.plot(dates,-vol, color = 'red')
        plt.plot(dates,vol, color = 'red')
        plt.title(title)
        plt.show()

    def get_realized_volatility_forecast(
            self, 
            risk_factor: RiskFactor, 
            reference_time: datetime) -> float: 
        delta_time = garch_time_delta
        instrument_name = risk_factor.base_currency.code + '-PERPETUAL'
        dt = 365*24
        ts = self.get_instrument_mark_price_time_serie(
            instrument_name,
            reference_time,
            delta_time) 
        egarch = ts.normal_garch_fit()
        params = egarch.params
        res, condsigmas = egarch.resid, egarch.conditional_volatility
        e, s = res[len(res)-1], condsigmas[len(condsigmas)-1]
        omega = params['omega'].item()
        alpha = params['alpha[1]'].item()
        beta = params['beta[1]'].item()
        variance = omega + alpha*np.abs(e)**2 + beta*s**2
        if np.isfinite(variance): 
            vol = np.sqrt(dt)*np.sqrt(variance)
            return vol.item()
        else: 
            return self.get_realized_volatility_forecast(
                risk_factor, 
                reference_time - timedelta(hours=1))
    
    def get_implied_volatility_log_change_forecast(
            self, 
            risk_factor: RiskFactor, 
            reference_time: datetime) -> float: 
        dt = auto_regressive_time_delta
        ts = self.get_risk_factor_atm_factor_time_serie(
            risk_factor, reference_time,dt)
        ar = ts.ar_12lag_fit()
        ivlc = ar.params['Const'].item()
        lr = ts.log_difference
        n = len(lr)
        for i in range(1,13): 
            name = 'y['+str(i)+']'
            r,a = lr[len(lr)-i], ar.params[name].item()
            ivlc = ivlc + r*a
        return ivlc.item()
        
    def get_date_vector_for_backtest(self) -> List[datetime]: 
        min_date = min(self.dates_dt)
        garch_dt = garch_time_delta
        ar_dt = auto_regressive_time_delta
        max_dt = max([garch_dt,ar_dt]) 
        min_date_for_bt = min_date + max_dt
        return [d for d in self.dates_dt if d>min_date_for_bt]
    
def get_market_loader() -> MarketLoader: 
    with open('data/market_loader_object.pkl', 'rb') as inp:
        return pickle.load(inp)
    
def update_market_loader(nstart:int=0, nend:int=1000000) -> None: 
    obj = MarketLoader(nstart,nend)
    with open('data/market_loader_object.pkl', 'wb') as outp:  
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
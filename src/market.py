from dataclasses import dataclass
from typing import List
import pandas as pd 
import math 
from datetime import datetime, timedelta
import numpy as np 
import scipy.interpolate
import matplotlib.pyplot as plt 
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
from src.quant.ssvi import SSVI, CalibrateSSVI
from src.quant.timeserie import TimeSerie
from src.tools import update_dict_of_list

@dataclass
class VolatilitySurface: 
    reference_time : datetime 
    risk_factor : RiskFactor
    ssvi : SSVI

@dataclass
class FutureTermStructure: 
    reference_time : datetime 
    risk_factor : RiskFactor
    datamap : dict[float, float]

    def __post_init__(self): 
        self.datamap = dict(sorted(self.datamap.items()))
    
    def future_price(self, t:float): 
        t_vec = np.array(list(self.datamap.keys())) 
        max_t_vec = np.max(t_vec)
        t = min(t, max_t_vec)
        v = np.array(list(self.datamap.values())) 
        interp = scipy.interpolate.interp1d(t_vec,v)
        return interp(t)

@dataclass
class Market: 
    reference_time : datetime 
    risk_factor : RiskFactor
    quotes : List[InstrumentQuote]
    spot : Spot 
    perpetual : PerpetualFuture
    futures : List[Future]
    options : List[Option]

    def __post_init__(self): 
        self._set_option_data()
        self.liquid_options = self.get_liquid_option()
        self._set_advanced_option_data()
        self.atmtvarmap = self.get_atmtvar_map()
        self.atm_factor = self.get_atm_factor()
    
    def _set_option_data(self) -> None: 
        self.option_expiries = list()
        self.option_names = list()
        self.put_options_name = list()
        self.call_options_name = list()
        self.mapped_expiries_option_names = dict()
        for o in self.options: 
            self.option_names.append(o.name)
            if o.expiry_dt not in self.option_expiries:
                self.option_expiries.append(o.expiry_dt)
            if o.call_or_put == 'C': 
                self.call_options_name.append(o.name)
            else: self.put_options_name.append(o.name)
            self.mapped_expiries_option_names = update_dict_of_list(
                o.expiry_dt,o.name,self.mapped_expiries_option_names)
    
    def _set_advanced_option_data(self) -> None: 
        self.mapped_liquid_strikes = dict()
        self.mapped_liquid_exptime = dict()
        self.mapped_liquid_futprice = dict()
        self.mapped_liquid_ivquote = dict()
        fut_terms = self.get_future_term_structure()
        for o in self.liquid_options: 
            quote = self.get_quote(o.name)
            expiry = o.expiry_dt
            t = o.time_to_expiry(self.reference_time)
            price = fut_terms.future_price(t)
            iv = quote.mid_iv/100
            self.mapped_liquid_strikes = update_dict_of_list(
                expiry,o.strike,self.mapped_liquid_strikes)
            self.mapped_liquid_exptime = update_dict_of_list(
                expiry,t,self.mapped_liquid_exptime)
            self.mapped_liquid_futprice = update_dict_of_list(
                expiry,price,self.mapped_liquid_futprice)
            self.mapped_liquid_ivquote = update_dict_of_list(
                expiry,iv,self.mapped_liquid_ivquote)
            
    def get_future_term_structure(self) -> FutureTermStructure: 
        futname = [f.name for f in self.futures]
        futexp = [f.time_to_expiry(self.reference_time) for f in self.futures]
        filterquotes = [q for q in self.quotes if q.instrument_name in futname]
        futprices = [q.order_book.mark_price for q in filterquotes]
        futexp.insert(0,0)
        perp_quote = self.get_quote(self.perpetual.name)
        futprices.insert(0,perp_quote.order_book.mark_price)
        datamap = dict(zip(futexp,futprices))
        return FutureTermStructure(self.reference_time,self.risk_factor,datamap)
    
    def get_liquid_option(self, n = 5) -> List[Option]: 
        filtered_names = list()
        quotes = [q for q in self.quotes 
                  if q.instrument_name in self.option_names
                  and not np.isnan(q.mid_iv)
                  and not np.isnan(q.order_book.spread)
                  and q.order_book.spread > 0
                  and q.ask_iv>0
                  and q.bid_iv>0]
        for e in self.option_expiries: 
            opt_name = self.mapped_expiries_option_names[e]
            put_spreads = {q.instrument_name: q.order_book.spread for q in quotes
                           if q.instrument_name in opt_name 
                           and q.instrument_name in self.put_options_name}
            call_spreads = {q.instrument_name: q.order_book.spread for q in quotes
                            if q.instrument_name in opt_name 
                            and q.instrument_name in self.call_options_name}
            put_spreads = dict(sorted(
                put_spreads.items(), 
                key=lambda item: item[1]))
            call_spreads = dict(sorted(
                call_spreads.items(), 
                key=lambda item: item[1]))
            if len(put_spreads)>=n: filt_put_names = list(put_spreads.keys())[0:n]
            else: filt_put_names = list(put_spreads.keys())
            if len(call_spreads)>=n: filt_call_names = list(call_spreads.keys())[0:n]
            else: filt_call_names = list(call_spreads.keys())
            filtered_names = filtered_names+filt_put_names
            filtered_names = filtered_names+filt_call_names
        return [o for o in self.options if o.name in filtered_names]

    def get_atm_factor(self) -> float: 
        t_vec = np.array(list(self.atmtvarmap.keys()))
        v = np.array(list(self.atmtvarmap.values()))
        l = scipy.stats.linregress(t_vec, v)
        return np.sqrt(l.slope.item())
        
    def get_quote(self, instrument_name:str) -> InstrumentQuote: 
        return [q for q in self.quotes if q.instrument_name==instrument_name][0]

    def get_atmtvar_map(self) -> dict[float, float]: 
        output = dict()
        output[0] = 0
        time_to_expiry_limit = 5*60*60/(365*24*60*60)
        for e in list(self.mapped_liquid_strikes.keys()): 
            t_vec = self.mapped_liquid_exptime[e]
            t = t_vec[0]
            if t<time_to_expiry_limit: 
                continue
            K = self.mapped_liquid_strikes[e]
            F = self.mapped_liquid_futprice[e]
            k = np.log(np.array(K)/np.array(F))
            iv = self.mapped_liquid_ivquote[e]
            if len(iv)==1: 
                output[t] = iv
                continue
            k_iv_map = dict(zip(k,iv))
            k_iv_map = dict(sorted(k_iv_map.items()))
            k = list(k_iv_map.keys())
            iv = list(k_iv_map.values())
            if np.all(np.array(k)>0): 
                output[t] = k_iv_map[min(k)]
                continue 
            if np.all(np.array(k)<0): 
                output[t] = k_iv_map[max(k)]
                continue 
            interp = scipy.interpolate.interp1d(np.array(k), np.array(iv))
            output[t] = t*(interp(0).item())**2
        return dict(sorted(output.items())) 
    
    def get_volatility_surface(self) -> VolatilitySurface: 
        k, sigma, t = list(), list(), list()
        for e in list(self.mapped_liquid_strikes.keys()): 
            K = self.mapped_liquid_strikes[e]
            F = self.mapped_liquid_futprice[e]
            k = k + np.log(np.array(K)/np.array(F))
            t = t + self.mapped_liquid_exptime[e]
            sigma = sigma + self.mapped_liquid_ivquote[e]
        ssvicalibrator = CalibrateSSVI(
            np.array(sigma),
            k,
            np.array(t),
            self.atm_tvar_map)
        return VolatilitySurface(
            self.reference_time,
            self.risk_factor,
            ssvicalibrator.calibrate())

    def check_ssvi_fit(self) -> None: 
        volsurface = self.get_volatility_surface()
        ssvi = volsurface.ssvi
        fut_terms = self.get_future_term_structure()
        expiries = list(set([o.expiry_dt for o in self.liquid_options]))
        for e in expiries: 
            opt = [o for o in self.liquid_options if o.expiry_dt==e]
            opt_names = [o.name for o in opt]
            quotes = [self.get_quote(n) for n in opt_names]
            t = opt[0].time_to_expiry(self.reference_time)
            K = [o.strike for o in opt]
            F = fut_terms.future_price(t).item()
            k = np.log(np.array(K)/F)
            iv = [q.mid_iv/100 for q in quotes]
            k_iv_map = dict(zip(k,iv))
            k_iv_map = dict(sorted(k_iv_map.items()))
            k = list(k_iv_map.keys())
            iv = list(k_iv_map.values())
            plt.plot(k, iv)
            plt.plot(k, ssvi.implied_volatility(np.array(k),t))
            plt.legend(['Market', 'SSVI'])
            plt.title(self.risk_factor.code + ' : ' + str(t))
            plt.show()
     
class MarketLoader: 
    def __init__(self, n:int, do_market_processing:bool = True) -> None:
        self.n, self.do_market_processing = n, do_market_processing
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
        data = pd.read_csv('data/aggregate_deribit_data.csv', nrows=self.n)
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
        if not self.do_market_processing: return list()
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
        egarch = ts.skewed_student_egarch_fit()
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
        instrument_name = risk_factor.base_currency.code + '-PERPETUAL'
        garch_time_delta = timedelta(days = 20)
        dt = 365*24
        ts = self.get_instrument_mark_price_time_serie(
            instrument_name,
            reference_time,
            garch_time_delta) 
        egarch = ts.skewed_student_egarch_fit()
        params = egarch.params
        res, condsigmas = egarch.resid, egarch.conditional_volatility
        e, s = res[len(res)-1], condsigmas[len(condsigmas)-1]
        omega = params['omega'].item()
        alpha = params['alpha[1]'].item()
        beta = params['beta[1]'].item()
        _gamma = params['gamma[1]'].item()
        variance = np.exp(omega + alpha*(np.abs(e/s) - np.sqrt(2/np.pi))
        +_gamma*(e/s)+beta*np.log(s**2))
        vol = np.sqrt(dt)*np.sqrt(variance)
        return vol.item()
    
    def get_implied_volatility_log_change_forecast(
            self, 
            risk_factor: RiskFactor, 
            reference_time: datetime) -> float: 
        ar_time_delta = timedelta(days = 20)
        ts = self.get_risk_factor_atm_factor_time_serie(
            risk_factor, reference_time,ar_time_delta)
        ar = ts.ar_12lag_fit()
        ivlc = ar.params['Const'].item()
        lr = ts.log_difference
        n = len(lr)
        for i in range(1,13): 
            name = 'y['+str(i)+']'
            r,a = lr[len(lr)-i], ar.params[name].item()
            ivlc = ivlc + r*a
        return ivlc.item()
        

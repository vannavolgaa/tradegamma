from typing import List
from datetime import datetime
from src.loader import get_market_loader
from src.trader import VolatilityTrader, Portfolio, CashFlow, Trade, Position, VTBook, Currency


def backtest_volatility_trader() -> Portfolio: 
    loader = get_market_loader()
    deposit = CashFlow(100000, loader.usd)
    exposure = 0.1 
    dates = loader.get_date_vector_for_backtest()
    date0 = dates[0]
    risk_factor = loader.btcusd
    market = [m for m in loader.markets 
          if m.risk_factor==risk_factor
          and m.reference_time==date0][0]
    spot_instrument = market.spot 
    spot_quote = market.get_quote(spot_instrument.name)
    quantity_to_trade = deposit.amount/spot_quote.order_book.best_ask
    initial_trade = Trade(spot_instrument, quantity_to_trade,
                        spot_quote.order_book.best_ask,
                        loader.usd,date0)
    book = VTBook(list(), [initial_trade])
    output = dict()
    i = 0
    for d in dates: 
        print(d)
        iv_forecast = loader.get_implied_volatility_log_change_forecast(
            risk_factor, d)
        re_forecast = loader.get_realized_volatility_forecast(risk_factor,d)
        market = [m for m in loader.markets 
          if m.risk_factor==risk_factor
          and m.reference_time==d][0]
        trader = VolatilityTrader(
            book, market,exposure,iv_forecast,re_forecast,deposit)
        book = trader.update_book()
        pft = book.to_portfolio(market)
        output[d] = pft.get_usd_value(deposit)
        print(output[d])
        i = i +1
        perc = round(100*i/len(dates),2)
        print(str(perc) + '% of the backtest')
    print('Backtest is done.')
    return output



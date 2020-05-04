"""
Code from: https://github.com/justinfay
"""
from time import sleep
import json
import datetime
import requests
import pandas as pd


BIN_SIZES = ['1m', '5m', '1h', '1d']
SYMBOLS = ['XBTUSD']
BUCKETED_URL = 'https://www.bitmex.com/api/v1/trade/bucketed'


# @cache.memoize()
def get_bars(
    bin_size=BIN_SIZES[0],
    symbol=None,
    count=100,
    start=None,
    start_time=None,
    end_time=None,
):
    print('making bitmex http request')
    if start_time is not None:
        start_time = start_time.isoformat()[:-7]
    if end_time is not None:
        end_time = end_time.isoformat()[:-7]
    resp = requests.get(
        BUCKETED_URL,
        params={
            'binSize': bin_size,
            'symbol': symbol,
            'count': count,
            'start': start,
            'startTime': start_time,
            'endTime': end_time,
        }
    )
    try:
        resp.raise_for_status()
    except:
        raise
    return resp.json()

if __name__ == "__main__":

    bars = []
    time = datetime.datetime.now() - datetime.timedelta(hours=24*7*30)

    while time < datetime.datetime.now():
        bars.extend(get_bars(
            bin_size='1m',
            symbol=SYMBOLS[0],
            start_time=time,
            count=200))
        time += datetime.timedelta(minutes=200)
        sleep(5)

        with open(f'bitmex/data/{datetime.datetime.now()}.json', 'w') as fh:
            json.dump(bars, fh)

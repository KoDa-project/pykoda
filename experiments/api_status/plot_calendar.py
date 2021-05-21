import requests
import multiprocessing

import pandas as pd
import tqdm

import matplotlib.pyplot as plt
import calplot


def _get_content(url):
    with requests.get(url, stream=True) as req:
        content = req.raw.read(3)
    return content

def get_status_data(feed, company, start_date, end_date):
    start_date = start_date.replace('_', '-')
    end_date = end_date.replace('_', '-')


    feed_operative = []
    dates = []
    for datetime in tqdm.tqdm(pd.date_range(start_date, end_date, freq='d')):
        date, time_code = str(datetime).split()
        url = f'https://koda.linkoping-ri.se/KoDa/api/v0.1?company={company}&feed={feed}&date={date}'

        pool = multiprocessing.Pool(processes=1)
        async_result = pool.apply_async(_get_content, (url,))
        pool.close()
        try:
            content = async_result.get(1)
        except multiprocessing.context.TimeoutError:
            pool.terminate()
            continue

        if content == b'BZh' or content.startswith(b'PK'):
            feed_operative.append(1.)
        else:
            feed_operative.append(-1)
        dates.append(datetime)

    return pd.Series(feed_operative, index=dates)


for company in ('otraf', 'sl', 'dintur'):
    for feed in ('TripUpdates', 'VehiclePositions', 'GTFSStatic'):
        df = get_status_data(feed, company, '2020_01_01', '2020_10_30')
        calplot.calplot(df, cmap='bwr_r', suptitle=' '.join((company, feed, 'status')))
        plt.savefig(f'status_{company}_{feed}.png')
plt.show()

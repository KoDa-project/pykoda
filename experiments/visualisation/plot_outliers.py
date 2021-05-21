import numpy as np
import matplotlib.pyplot as plt

import pykoda


def plot_outliers(company, date, n_sigma: float = 5.0):
    """Plot stations that accumulate significant delays, defined as the ones that accumulate a median delay
    n_sigma times the median delay."""
    df = pykoda.datautils.get_data_range(feed='TripUpdates', company=company, start_date=date,
                                         start_hour=9, end_hour=23, merge_static=True)

    departure_threshold = n_sigma * np.nanmedian(df.departure_delay.values)
    arrival_threshold = n_sigma * np.nanmedian(df.arrival_delay.values)

    df_dep = df.groupby('stop_id').median().query('departure_delay > @departure_threshold')
    df_arr = df.groupby('stop_id').median().query('arrival_delay > @arrival_threshold')

    pykoda.plotutils.setup_mpl()

    fig, ax = plt.subplots(2, 1, subplot_kw=dict(projection=pykoda.geoutils.SWEREF99), sharex=True, sharey=True,
                           figsize=(6, 1.6 * 6))

    plt.sca(ax[0])
    plt.scatter(df_dep.stop_lon, df_dep.stop_lat, c=df_dep.departure_delay / 60, vmin=0)
    plt.title('Stations with delayed departures')
    plt.colorbar(label='Delay [m]')

    plt.sca(ax[1])
    plt.scatter(df_arr.stop_lon, df_arr.stop_lat, c=df_arr.arrival_delay / 60, vmin=0)
    plt.colorbar(label='Delay [m]')
    plt.title('Stations with delayed arrivals')

    # Add base maps
    pykoda.plotutils.add_basemap(9, ax[0])
    pykoda.plotutils.add_basemap(9, ax[1])


if __name__ == '__main__':
    COMPANY = 'otraf'
    DATE = '2020_08_21'
    plot_outliers(COMPANY, DATE)
    plt.show()

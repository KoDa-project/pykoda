import pandas as pd
import matplotlib.pyplot as plt

import pykoda


def plot_routes():
    COMPANY = 'sl'
    START_DATE = '2020_09_21'
    END_DATE = '2020_09_27'
    df_trip = pykoda.datautils.get_data_range(feed='TripUpdates', company=COMPANY, start_date=START_DATE, start_hour=0,
                                              end_date=END_DATE, end_hour=23, merge_static=True,
                                              query='route_short_name == "6"')

    pykoda.plotutils.setup_mpl()

    print(df_trip.head())

    fig, ax = plt.subplots(subplot_kw=dict(projection=pykoda.geoutils.SWEREF99), figsize=(10, 10))
    pykoda.plotutils.add_basemap(14, ax)

    df_stops = df_trip.groupby('stop_id').mean()
    max_delay = df_stops.arrival_delay.abs().max() / 60
    sc = plt.scatter(df_stops.stop_lon, df_stops.stop_lat, c=df_stops.arrival_delay / 60,
                     transform=pykoda.geoutils.PLATE_CARREE, zorder=2, cmap='bwr', vmin=-max_delay, vmax=max_delay)
    plt.title(f'All trips for route ')
    plt.colorbar(sc, label='Average stop delay [m]')

    _, ax2 = plt.subplots()
    sc = ax2.scatter(df_trip.datetime, df_trip.shape_dist_traveled / 1e3, c=df_trip.arrival_delay / 60,
                     alpha=0.7, cmap='bwr', vmin=-max_delay, vmax=max_delay)

    plt.title('Arrival delays')
    plt.ylabel('Travelled distance [km]')
    plt.xlabel('Time')
    plt.xlim(df_trip.datetime.min(), df_trip.datetime.max())
    plt.colorbar(sc, label='Delay [m]')


if __name__ == '__main__':
    plot_routes()
    plt.savefig('a.png')
    plt.close()
    plt.savefig('b.png')
    plt.show()

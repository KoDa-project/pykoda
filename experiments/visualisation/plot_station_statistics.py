import numpy as np
import matplotlib.pyplot as plt

import pykoda


def _adjust_violin(violin):
    for viol in violin['bodies']:
        viol.set_facecolor('#D43F3A')
        viol.set_edgecolor('#D43F3A')
        viol.set_alpha(0.3)
    for key in violin.keys():
        if key == 'bodies':
            continue
        viol = violin[key]
        viol.set_alpha(0.4)
        viol.set_color('#D43F3A')


def plot_station_statistics(company, date):
    """Plot some statistics on delays for every station.

    Figure 1 overlays the histograms of delays for every stop during the day.

    Figure 2 generates a violin plot of the same data, where the position of each violin
    is the distance to the city centre, as defined as the median latitude and longitude of all the stops in the data.

    The distance is computed using the flat Earth approximation, and illustrates how to use pykoda.geoutils.
    """

    df = pykoda.datautils.get_data_range(feed='TripUpdates', company=company, start_date=date,
                                         start_hour=0, end_hour=23, merge_static=True)

    stop_grouped = df.groupby('stop_id')
    pykoda.plotutils.setup_mpl()

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    for _, group in stop_grouped:
        plt.sca(ax[0])
        plt.hist(group.departure_delay.dropna() / 60, histtype='stepfilled', color='b', alpha=0.1, bins='auto',
                 range=(-20, 20))
        plt.sca(ax[1])
        plt.hist(group.arrival_delay.dropna() / 60, histtype='stepfilled', color='g', alpha=0.1, bins='auto',
                 range=(-20, 20))

    plt.sca(ax[0])
    plt.title('Departure delays')
    plt.axvline(0, color='k', ls='-.')
    plt.sca(ax[1])
    plt.title('Arrival delays')
    plt.axvline(0, color='k', ls='-.')
    plt.xlabel('Delay [m]')

    fig, axes = plt.subplots(2, 1, sharex=True)
    centre = stop_grouped.stop_lat.mean().median(), stop_grouped.stop_lon.mean().median()

    positions = []
    data_arrival = []
    data_depart = []

    for _, group in stop_grouped:
        if len(group) < 100:
            continue
        pos = group.stop_lat.mean(), group.stop_lon.mean()
        distance = pykoda.geoutils.flat_distance(pos, centre)

        data_depart.append(group.departure_delay.dropna() / 60)
        data_arrival.append(group.arrival_delay.dropna() / 60)
        positions.append(distance)

    positions = np.array(positions)
    widths = 0.5 * positions.ptp() / len(positions) * positions / 1e3

    violin = axes[0].violinplot(data_depart, positions, widths=widths, showmedians=True)
    _adjust_violin(violin)

    violin = axes[1].violinplot(data_arrival, positions, widths=widths,
                                showmedians=True)
    _adjust_violin(violin)

    plt.sca(axes[0])
    plt.title('Departure delays')
    plt.axhline(0, color='k', alpha=0.2)
    plt.ylabel('Delays by station [m]')
    plt.sca(axes[1])
    plt.title('Arrival delays')
    plt.axhline(0, color='k', alpha=0.2)
    plt.semilogx()
    plt.ylabel('Delays by station [m]')
    plt.xlabel('Distance to the city centre')


if __name__ == '__main__':
    COMPANY = 'otraf'
    DATE = '2020_08_21'
    plot_station_statistics(COMPANY, DATE)
    plt.show()

import random

import matplotlib.pyplot as plt

import pykoda


def plot_routes(company, date, route=None):
    """Plot the trajetories taken by every vehicle across a given route, and the delays in departure and arrival.

    On the map, the trajectories are shown in blue and the stops are marked with dots,
    coloured according to the average delay accumulated at that position.

    If route is not given, one will be selected at random.
    """
    df_trip = pykoda.datautils.get_data_range(feed='TripUpdates', company=company, start_date=date,
                                              start_hour=0, end_hour=23, merge_static=True)

    # Select the route and query the data
    if route is None:
        route = random.choice(df_trip.route_id.values)
    df_trip = df_trip.query('route_id == @route')

    # Infer the user-facing name of the route.
    # It can be either in the route_long_name or route_short_name field.
    long_names = df_trip.route_long_name.dropna().unique()
    if long_names.size > 0:
        route_name = ' '.join(long_names)
    else:
        short_names = df_trip.route_short_name.dropna().unique()
        route_name = ' '.join(short_names)

    # Obtain the corresponding trip_ids
    trip_ids = df_trip.trip_id.unique()

    # Load vehcile data
    df_vehicle = pykoda.datautils.get_data_range(feed='VehiclePositions', company=company, start_date=date,
                                                 start_hour=9, end_hour=23, query='trip_id in @trip_ids')

    pykoda.plotutils.setup_mpl()

    fig, ax = plt.subplots(subplot_kw=dict(projection=pykoda.geoutils.SWEREF99), figsize=(10, 10))
    pykoda.plotutils.add_basemap(14, ax)

    _, ax2 = plt.subplots()

    for i, (trip_id, trip) in enumerate(df_vehicle.groupby('trip_id')):
        trip.sort_values(by='timestamp', inplace=True)
        ax.plot(trip.vehicle_position_longitude, trip.vehicle_position_latitude, color='b', alpha=0.3,
                transform=pykoda.geoutils.PLATE_CARREE, zorder=1)

        trip_updates = df_trip.query('trip_id == @trip_id').sort_values(by='scheduled_departure_time')

        ax2.plot(trip_updates.shape_dist_traveled / 1e3, trip_updates.arrival_delay / 60, color='b', marker='^',
                 alpha=0.4, label='Arrival' if i == 0 else None)
        ax2.plot(trip_updates.shape_dist_traveled / 1e3, trip_updates.arrival_delay / 60, color='g', marker='v',
                 alpha=0.4, label='Departure' if i == 0 else None)

    plt.title('Delays at stops')
    plt.xlabel('Travelled distance [km]')
    plt.ylabel('Delay [m]')
    plt.axhline(0, color='k')
    plt.legend(loc=0)

    plt.sca(ax)
    df_stops = df_trip.groupby('stop_id').mean()
    max_delay = df_stops.arrival_delay.abs().max() / 60
    sc = plt.scatter(df_stops.stop_lon, df_stops.stop_lat, c=df_stops.arrival_delay / 60,
                     transform=pykoda.geoutils.PLATE_CARREE, zorder=2, cmap='bwr', vmin=-max_delay, vmax=max_delay)
    plt.title(f'All trips for route {route_name}')
    plt.colorbar(sc, label='Average stop delay [m]')


if __name__ == '__main__':
    COMPANY = 'otraf'
    DATE = '2020_08_21'
    plot_routes(COMPANY, DATE)
    plt.show()

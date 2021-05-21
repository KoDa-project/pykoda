import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs

import tqdm

import pykoda


def plot_geo_delays(company, date, start_hour, end_hour, max_delay=20):
    """Plot an animated map of the average delay for every stop over time."""
    pykoda.plotutils.setup_mpl()

    data = pykoda.datautils.get_data_range('TripUpdates', company, date, start_hour=start_hour, end_hour=end_hour,
                                           merge_static=True)

    # Extract the hour for grouping
    data['hour'] = data.scheduled_arrival_time.dt.hour

    # Save the data for each station, direction, and hour and store them in a list.
    frames = []
    for h, df in tqdm.tqdm(data.groupby('hour')):
        df = df.groupby(['stop_id', 'direction_id']).mean()
        frames.append((df.stop_lat, df.stop_lon, df.arrival_delay / 60))

    # Setting up the figure, including limits and base map.
    fig, ax = plt.subplots(subplot_kw=dict(projection=pykoda.geoutils.SWEREF99))
    sc = plt.scatter([], [], c=[], cmap='bwr', vmin=-max_delay, vmax=max_delay, s=3)
    plt.colorbar(sc, label='Delay [m]')

    if company == 'otraf':
        ax.set_extent((14.5, 17, 57.7, 59))

    pykoda.plotutils.add_basemap(9, ax)

    # Now, draw the data for each frame in an animation and save.
    def run(i):
        del fig.axes[0].collections[-1]
        title = fig.axes[0].set_title(f'Delay plot for {date} {start_hour + i} h')
        lat, lon, delay = frames[i]
        sc = plt.scatter(lon, lat, c=delay, cmap='bwr', vmin=-max_delay,
                         vmax=max_delay, s=3, transform=ccrs.PlateCarree())
        return sc, title

    ani = animation.FuncAnimation(fig, run, frames=len(frames))
    ani.save('delays_map.mp4')
    ani.save("delays_map.gif", writer="imagemagick")

    # Same thing, but now zoomed in.
    fig, ax = plt.subplots(subplot_kw=dict(projection=pykoda.geoutils.SWEREF99))
    sc = plt.scatter([], [], c=[], cmap='bwr', vmin=-max_delay, vmax=max_delay, s=3)
    plt.colorbar(sc, label='Delay [m]')
    if company == 'otraf':
        ax.set_extent((15.56, 15.72, 58.37, 58.44))
    pykoda.plotutils.add_basemap(12, ax)

    ani = animation.FuncAnimation(fig, run, frames=len(frames))
    ani.save('delays_map_zoom.mp4')
    ani.save("delays_map_zoom.gif", writer="imagemagick")


if __name__ == '__main__':
    START_HOUR = 9
    END_HOUR = 20
    COMPANY_NAME = 'otraf'
    DATE = '2020-08-21'

    plot_geo_delays(COMPANY_NAME, DATE, START_HOUR, END_HOUR)
    plt.show()

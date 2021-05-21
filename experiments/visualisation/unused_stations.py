import numpy as np

import matplotlib.pyplot as plt

import pykoda


def main(date, company):
    # Get the coordinates from the static data
    static_data = pykoda.load_static_data(date=date, company=company, remove_unused_stations=False)

    routes = set(static_data.stop_times.index.levels[1])
    st = static_data.stops.query('stop_id not in @routes')
    coordinates_alt = np.stack([st.stop_lon.values, st.stop_lat.values, ], axis=-1)
    plt.scatter(*coordinates_alt.T, s=3, color='r',  label='Unused stations')

    st = static_data.stops.query('stop_id in @routes')
    coordinates_alt = np.stack([st.stop_lon.values, st.stop_lat.values], axis=-1)
    plt.scatter(*coordinates_alt.T, s=3, color='b', alpha=0.3, label='Stations on routes')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    company = 'otraf'
    date = '2020_08_21'
    main(date, company)

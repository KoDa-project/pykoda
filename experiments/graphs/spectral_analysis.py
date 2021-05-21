import numpy as np
import networkx as nx
from scipy import spatial

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import pykoda

import tqdm

"""
In this example we are going to divide Stockholm in zones , and create a connectivity network between zones. We will
then perform spectral analysis to see what kind of information it can reveal.

The zones are defined by fast transit stations (trains such as pendeltåg and subway), and the boundaries are defined by
geographical proximity.
"""


def get_graph(company: str, date: str, hour: int):
    static_data = pykoda.datautils.load_static_data(company, date, remove_unused_stations=True)

    # Extract the stations corresponding to (Roslagsbanan, Saltsjöbanan, tunnelbannan), (pendeltåg, pendelbåt)
    # We jump from routes to trips to actual stops.
    _query = ' or '.join('(route_desc.str.contains("{}"))'.format(mode) for mode in ('banan', 'endel'))

    special_routes = static_data.routes.query(_query, engine='python').route_id.unique()
    special_trips = static_data.trips.query('route_id in @special_routes').eval('trip_id', engine='python').unique()
    _stop_query = 'trip_id in @special_trips and departure_time.dt.hour == @hour'
    special_stops = static_data.stop_times.query(_stop_query).eval('stop_id', engine='python').unique()

    special_stops_data = static_data.stops.query('stop_id in @special_stops').copy()

    # Prune with a bounding box.
    eps_lon = 0.05 * np.ptp(special_stops_data.stop_lon.dropna())
    eps_lat = 0.05 * np.ptp(special_stops_data.stop_lat.dropna())
    lon_min, lon_max = special_stops_data.stop_lon.min() - eps_lon, special_stops_data.stop_lon.max() + eps_lon
    lat_min, lat_max = special_stops_data.stop_lat.min() - eps_lat, special_stops_data.stop_lat.max() + eps_lat

    all_stops_data = static_data.stops.query(
        '(@lon_min <= stop_lon <= @lon_max) & (@lat_min <= stop_lat <= @lat_max)').copy()

    # We now project the coordinates so that Euclidean distances are representative:
    special_stops_coordinates = np.stack([special_stops_data.stop_lon, special_stops_data.stop_lat], axis=1)
    projected_coordinates = pykoda.geoutils.project_points(special_stops_coordinates)
    all_stops_coordinates = np.stack([all_stops_data.stop_lon, all_stops_data.stop_lat], axis=1)
    all_stops_projected = pykoda.geoutils.project_points(all_stops_coordinates)

    # Assign Voronoi regions
    kdtree = spatial.cKDTree(projected_coordinates)
    dist, ids = kdtree.query(all_stops_projected, n_jobs=-1)

    # Save the vales in the data frames
    all_stops_data['zone'] = ids
    all_stops_data['distance'] = dist
    all_stops_data['x'] = all_stops_coordinates[:, 0]
    all_stops_data['y'] = all_stops_coordinates[:, 1]

    special_stops_data['zone'] = kdtree.query(projected_coordinates)[1]
    special_stops_data['x'] = projected_coordinates[:, 0]
    special_stops_data['y'] = projected_coordinates[:, 1]

    # Build the graph
    G = build_graph(static_data, all_stops_data, ids, hour)

    return G, static_data, all_stops_data, special_stops_data


def build_graph(static_data, all_stops, ids, hour, with_selfloops=True) -> nx.MultiDiGraph:
    stop_to_zone = dict(zip(all_stops.eval('stop_id', engine='python'), ids))

    # And build the graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(set(stop_to_zone.values()))

    # Add an edge between corresponding zones each trip that connects two stations.
    _query = 'stop_id in @stop_to_zone and departure_time.dt.hour == @hour'
    for _, route in tqdm.tqdm(static_data.stop_times.query(_query).groupby('trip_id')):
        stations = route.sort_values(by='stop_sequence').eval('stop_id', engine='python').values
        zones = [stop_to_zone[st] for st in stations]
        if not with_selfloops:
            zones = np.array(zones)
            zones = zones[np.r_[True, zones[:-1] != zones[1:]]]
        for i in range(len(stations) - 1):
            G.add_edge(zones[i], zones[i + 1])
    return G


def spectral_graph_analysis(company, date):
    G, static_data, all_stops_data, special_stops_data = get_graph(company, date, 9)

    # Now we can compute a few graph metrics
    centrality = nx.eigenvector_centrality_numpy(G)
    pagerank = nx.pagerank_scipy(G)
    spectrum = dict(zip(G.nodes, np.abs(nx.adjacency_spectrum(G))))
    modularity_spectrum = dict(zip(G.nodes, np.abs(nx.modularity_spectrum(nx.DiGraph(G)))))

    measures = [centrality, pagerank, spectrum, modularity_spectrum]
    labels = 'Eigenvector centrality', 'PageRank', 'Adjacency spectrum', 'Modularity spectrum'

    # And plot them in the same figure
    def plot_measure(data, measure, ax, name):
        values = [measure[n] for n in data.zone]
        sc = ax.scatter(data.stop_lon, data.stop_lat, c=values, norm=colors.LogNorm(), s=4, alpha=0.5,
                        transform=pykoda.geoutils.PLATE_CARREE)
        plt.colorbar(sc, ax=ax)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        # This seems to be hogging memory and taking forever. Not sure if it is an OSM server problem,
        # or something odd with this plot.

        ## pykoda.plotutils.add_basemap(ax=ax)
        pykoda.plotutils.add_water(ax)


    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10),
                             subplot_kw={'projection': pykoda.geoutils.SWEREF99})
    plt.suptitle('Graph measures on major zones')
    for meas, ax, lb in zip(measures, axes.flatten(), labels):
        plot_measure(special_stops_data, meas, ax, lb)


if __name__ == '__main__':
    company = 'sl'
    date = '2020_09_24'

    pykoda.plotutils.setup_mpl()
    spectral_graph_analysis(company, date)

    plt.show()

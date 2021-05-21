import pprint

import numpy as np

try:
    import alphashape

    USE_ALPHA = True
except ImportError:
    from scipy import spatial
    from shapely.geometry import Polygon

    USE_ALPHA = False

import osmnx as ox
import networkx as nx

import matplotlib.pyplot as plt

import pykoda


def main(date, company):
    # Get the coordinates from the static data
    static_data = pykoda.load_static_data(date=date, company=company, remove_unused_stations=True)

    # Remove isolated stations
    di_graph = pykoda.graphutils.build_nx_graph(static_data, merge_parent_stations=True)
    remove_edges = [(u, v,) for (u, v, d) in di_graph.edges(data=True) if d['distance'] > 10_000]
    di_graph.remove_edges_from(remove_edges)

    # Let's print some basic information of the graph
    print('Public transport graph statistics:')
    pprint.pprint(dict((k, v) for k, v in ox.basic_stats(di_graph).items()
                       # removing long dictionaries and missing values
                       if not isinstance(v, dict) and v and np.isfinite(v)))

    # Extract coordinates of the stations
    coordinates = []
    graph = di_graph.to_undirected()
    for nbunch in nx.connected_components(graph):
        if len(nbunch) > 50:
            coordinates.extend((graph.nodes[node]['lon'], graph.nodes[node]['lat']) for node in nbunch)
    coordinates = np.array(coordinates)

    # Extract a polygon that encloses the points.
    # The alpha-shape is a generalisation of the convex hull, allowing it to be more trimmed.

    if USE_ALPHA:
        alpha = 8.
        shape = alphashape.alphashape(coordinates, alpha)
    else:
        hull = spatial.ConvexHull(coordinates)
        shape = Polygon(hull.simplices)

    # Download the graph
    network_type = 'walk'  # or 'drive', 'bike', 'all'...
    graph = ox.graph_from_polygon(shape, clean_periphery=True, network_type=network_type)

    # Show some statistics
    print(f'OSM {network_type} graph statistics:')
    pprint.pprint(dict((k, v) for k, v in ox.basic_stats(graph).items()
                       # removing long dictionaries and missing values
                       if not isinstance(v, dict) and v and np.isfinite(v)))

    # Project it on UTM and plot
    ox.plot_graph(ox.project_graph(graph))
    plt.show()


if __name__ == '__main__':
    company = 'otraf'
    date = '2020_08_21'
    main(date, company)

import pykoda

import networkx as nx
import matplotlib.pyplot as plt


def plot_network(company, date, contract=True):
    """Draw a network of stops according to the static data.
    """
    static_data = pykoda.load_static_data(company, date)

    graph = pykoda.graphutils.build_nx_graph(static_data)

    # Remove isolated nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))

    if contract:
        graph = pykoda.graphutils.contract_nx_graph(graph)

    fig, ax = plt.subplots(subplot_kw=dict(projection=pykoda.geoutils.SWEREF99), figsize=(10, 10))
    pykoda.plotutils.add_basemap(8, ax)

    plt.scatter([data['lon'] for node, data in graph.nodes(data=True)],
                [data['lat'] for node, data in graph.nodes(data=True)], s=3, marker='s',
                transform=pykoda.geoutils.PLATE_CARREE)

    for n1, n2 in graph.edges():
        g1 = graph.nodes[n1]
        g2 = graph.nodes[n2]

        plt.plot([g1['lon'], g2['lon']], [g1['lat'], g2['lat']], color='k',
                 transform=pykoda.geoutils.PLATE_CARREE, alpha=0.5)

    plt.show()


if __name__ == '__main__':
    company = 'otraf'
    date = '2020_08_21'

    pykoda.plotutils.setup_mpl()
    plot_network(company, date, contract=True)

import statistics

import networkx as nx

from . import datautils, geoutils


def build_nx_graph(static_data: datautils.static_data, merge_parent_stations: bool = True) -> nx.MultiDiGraph:
    """Build a geographic graph from the stops in static data.

    The nodes are stops and the edges are routes connecting them.
    """

    if merge_parent_stations:
        stop_list = static_data.stops.query('parent_station.isnull()', engine='python')

        _stops = static_data.stops.query('~parent_station.isnull()', engine='python')
        mapping = dict(zip(_stops.index, _stops.parent_station))
    else:
        stop_list = static_data.stops

    nodes = dict()
    for ix, row in stop_list.iterrows():
        lat, lon = row.stop_lat, row.stop_lon
        x, y = geoutils.SWEREF99.transform_point(lon, lat, geoutils.PLATE_CARREE)  # Use the projection
        nodes[ix] = dict(lon=lon, lat=lat, type=row.location_type, platform_code=row.platform_code,
                         name=row.stop_name, x=x, y=y)

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes.items())

    for _, group in static_data.stop_times.groupby(['trip_id', ]):
        group = group.sort_values(by='stop_sequence')
        this_trip_stops = group.reset_index('stop_id').stop_id.values.tolist()
        path_lengths = group.shape_dist_traveled.tolist()

        if merge_parent_stations:
            this_trip_stops = [mapping.get(x, x) for x in this_trip_stops]

        for i in range(len(this_trip_stops) - 1):
            origin = this_trip_stops[i]
            destination = this_trip_stops[i + 1]

            origin_coord = nodes[origin]['lat'], nodes[origin]['lon']
            dest_coord = nodes[destination]['lat'], nodes[destination]['lon']
            distance = geoutils.flat_distance(origin_coord, dest_coord)

            path_length = path_lengths[i + 1] - path_lengths[i]
            G.add_edge(origin, destination, distance=distance, length=path_length, osmid='')

    G.graph['crs'] = 'SWEREF99'
    return G


def contract_nx_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """ Contract a route's graph replacing unbranched sequences with a single node.

    For example, if there are a series of sequential stops (s0..s3) connecting two hubs (H0, H1):

          > H0 -- s0 -- s1 -- s2 -- s3 -- H1 <

    They will be replaced by a single "path" node.

          > H0 -- path0 -- H1 <

    Single nodes connecting two hubs are replaced by a direct connection between the hubs:

         > H0 -- s0 -- H1 <

    will be transformed into:

        > H0 -- H1 <

    """
    graph = graph.copy()
    undirected = graph.to_undirected(reciprocal=False, as_view=False)

    undirected.remove_edges_from(nx.selfloop_edges(undirected))

    # These nodes are in simple paths, having two neighbours.
    nodes_in_paths = [node for node in undirected if undirected.degree(node) == 2]

    connected_components = list(nx.connected_components(undirected.subgraph(nodes_in_paths)))
    all_nodes = graph.nodes(data=True)
    i = 0
    for component in connected_components:
        if len(component) == 1:
            # This is a lone node between two hubs. Remove it by replacing its connections to the hubs themselves
            c = list(component)[0]
            nodes = undirected[c]

            # Check that we have nodes on both sides
            if len(nodes) == 1:
                continue
            n1, n2 = nodes
            if graph.has_edge(n1, c) or graph.has_edge(c, n2):
                graph.add_edge(n1, n2)
            if graph.has_edge(n2, c) or graph.has_edge(c, n1):
                graph.add_edge(n2, n1)

            # And remove it
            graph.remove_node(c)

        else:
            lat = statistics.mean(all_nodes[n]['lat'] for n in component)
            lon = statistics.mean(all_nodes[n]['lon'] for n in component)
            x = statistics.mean(all_nodes[n]['x'] for n in component)
            y = statistics.mean(all_nodes[n]['y'] for n in component)

            data = dict(lat=lat, lon=lon, x=x, y=y, name='', type='', platform_code='')
            new_node = f'path_{i}'
            graph.add_node(new_node, **data)

            for x in component:
                nx.contracted_nodes(graph, new_node, x, copy=False, self_loops=False)

            graph.remove_nodes_from(component)
            i += 1

    return graph

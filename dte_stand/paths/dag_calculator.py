import networkx
import itertools
from typing import Optional, Iterable
from collections import defaultdict
from contextlib import contextmanager
from networkx.algorithms.shortest_paths import shortest_path_length
from networkx.algorithms.traversal import dfs_tree, dfs_edges
from networkx.algorithms.dag import dag_longest_path
from typing import Generator

from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement

import logging
LOG = logging.getLogger(__name__)


class DAGCalculator(BasePathCalculator):
    def __init__(self, length_cutoff_fraction=1.5):
        super().__init__()
        self._forward_ordering = networkx.MultiDiGraph()
        self._reverse_ordering = networkx.MultiDiGraph()
        # how much longer the found paths can be compared to the shortest hop path
        self._length_cutoff = length_cutoff_fraction
        # keep a global track of nodes currently removed from topology
        self._forward_removed_nodes = set()
        self._reverse_removed_nodes = set()
        # precalculated path lengths to speed up the route construction
        self._forward_path_lengths: dict[str, dict[str, dict[tuple, int]]] = defaultdict(lambda: defaultdict(lambda: {}))
        self._reverse_path_lengths: dict[str, dict[str, dict[tuple, int]]] = defaultdict(lambda: defaultdict(lambda: {}))

    def _get_topology_edges_between_nodes(
                self, topology: networkx.MultiDiGraph,
                node1: str, node2: str) -> Generator[tuple[int, dict], None, None]:
        index = 0
        while True:
            try:
                yield index, topology.edges[node1, node2, index]
            except KeyError:
                # end of edges
                break
            index += 1

    def _node_with_longest_path(self, topology: networkx.Graph) -> str:
        max_path_length = 0
        max_len_node = None
        for node in topology.nodes():
            dag_topology = dfs_tree(topology, node)
            longest_path = dag_longest_path(dag_topology)
            new_length = len(longest_path)
            if new_length > max_path_length:
                max_path_length = new_length
                max_len_node = node
        return max_len_node

    def _save_path_length(self, graph: networkx.MultiDiGraph, source: str, destination: str,
                          removed_tuple: tuple) -> None:
        with self._hide_nodes(graph, removed_tuple):
            try:
                length = shortest_path_length(graph, source, destination)
            except networkx.NetworkXNoPath:
                return
            if graph is self._forward_ordering:
                self._forward_path_lengths[source][destination][removed_tuple] = length
            else:
                self._reverse_path_lengths[source][destination][removed_tuple] = length

    def _calculate_path_lengths(self, graph: networkx.MultiDiGraph) -> None:
        """
        Fill the self._forward_path_lengths and self._reverse_path_lengths with data

        Precalculate path lengths for each pair of nodes for each possible set of removed nodes.
        This is useful during path calculation when instead of doing shortest_path_length every time,
            we can look into this precalculated dict instead
        :param graph: either self._forward_ordering or self._reverse_ordering
        :return: None
        """
        all_nodes = list(graph.nodes())
        for source, destination in itertools.product(all_nodes, repeat=2):
            for removed1, removed2, removed3 in itertools.combinations(all_nodes, 3):
                removed_tuple = tuple(sorted((removed1, removed2, removed3)))
                if (source in removed_tuple) or (destination in removed_tuple):
                    continue
                self._save_path_length(graph, source, destination, removed_tuple)

            for removed1, removed2 in itertools.combinations(all_nodes, 2):
                removed_tuple = tuple(sorted((removed1, removed2)))
                if (source in removed_tuple) or (destination in removed_tuple):
                    continue
                self._save_path_length(graph, source, destination, removed_tuple)

            for removed_node in all_nodes:
                if (source == removed_node) or (destination == removed_node):
                    continue
                self._save_path_length(graph, source, destination, (removed_node,))

            self._save_path_length(graph, source, destination, tuple())

    def prepare_iteration(self, topology: networkx.MultiDiGraph) -> None:
        # TODO: if topology did not change, do not recalculate graphs
        #   or somehow pass the topology changes here
        self._forward_ordering.clear()
        self._reverse_ordering.clear()

        # make topology graph undirected to apply the st-numbering algorithm
        # all pair of links (A-B, B-A) will be treated as a single undirected link
        undirected_topo = networkx.Graph(topology)

        # use dfs to convert the graph into directed acyclic graph and this graph will be used by all nodes
        # how to choose source is a good question
        # for now we will take the node that has the longest path in the graph
        source_node = self._node_with_longest_path(undirected_topo)
        numbered_topo = self._dag_convert(undirected_topo, source_node)

        # using this dag, create two directed acyclic graphs from original topology
        # first graph is created according to dfs numbers
        # second graph is the first graph but with all edges' direction reverted (dfs are reverted accordingly)
        # all edges of the original topology are present either in first or in second graph
        max_number = len(numbered_topo.nodes) + 1
        for node_id, node_data in numbered_topo.nodes(data=True):
            # add node into forward ordering
            self._forward_ordering.add_node(node_id, **node_data)

            # add same node with reversed number into reverse ordering
            node_data['dfs_number'] = max_number - node_data['dfs_number']
            self._reverse_ordering.add_node(node_id, **node_data)

            # add links
            for _, neighbor_id in numbered_topo.edges(nbunch=node_id):
                # numbered_topo is a DiGraph without multi links
                # we need to look into original topology to get multi links

                # for each link (A, B) from numbered_topo, get all links (A, B, x) from original topology
                # and add them to forward ordering
                for edge_index, edge_data in self._get_topology_edges_between_nodes(topology, node_id, neighbor_id):
                    self._forward_ordering.add_edge(node_id, neighbor_id, key=edge_index, **edge_data)

                # for each link (A, B) from numbered_topo, get all links (B, A, x) from original topology
                # and add them to reverse ordering
                for edge_index, edge_data in self._get_topology_edges_between_nodes(topology, neighbor_id, node_id):
                    self._reverse_ordering.add_edge(neighbor_id, node_id, key=edge_index, **edge_data)

        self._calculate_path_lengths(self._forward_ordering)
        self._calculate_path_lengths(self._reverse_ordering)

    def _dag_convert(self, graph: networkx.Graph, s_node: str) -> networkx.DiGraph:
        # convert graph into directed acyclic graph using dfs search
        graph_nodes = graph.nodes(data=True)
        current_number = 1
        graph_nodes[s_node]['dfs_number'] = current_number
        for node_from, node_to in dfs_edges(graph, s_node):
            current_number += 1
            graph_nodes[node_to]['dfs_number'] = current_number

        dag_graph = networkx.DiGraph()
        dag_graph.add_nodes_from(graph_nodes)
        for node_from, node_to in graph.edges():
            if graph_nodes[node_from]['dfs_number'] < graph_nodes[node_to]['dfs_number']:
                dag_graph.add_edge(node_from, node_to)
            else:
                dag_graph.add_edge(node_to, node_from)

        return dag_graph

    @contextmanager
    def _hide_nodes(self, graph: networkx.MultiDiGraph, nodes: Iterable[str]) -> None:
        nodes_data = {}
        edges = []
        for node in nodes:
            try:
                nodes_data[node] = graph.nodes[node]
            except KeyError:
                continue
            edges.extend(list(graph.out_edges(nbunch=node, data=True, keys=True)))
            edges.extend(list(graph.in_edges(nbunch=node, data=True, keys=True)))
            graph.remove_node(node)

        if graph is self._forward_ordering:
            self._forward_removed_nodes.update(nodes)
        else:
            self._reverse_removed_nodes.update(nodes)

        try:
            yield
        finally:
            for node, node_data in nodes_data.items():
                graph.add_node(node, **node_data)
            graph.add_edges_from(edges)

            if graph is self._forward_ordering:
                self._forward_removed_nodes.difference_update(nodes)
            else:
                self._reverse_removed_nodes.difference_update(nodes)

    def _find_possible_nexthops(self, topology: networkx.MultiDiGraph,
                                source: str, destination: str, original_length: int) -> list[GraphPathElement]:
        """
        look at all nexthops a source has and return those nexthope where a simple path exists to destination
            original_length and self.length_cutoff is used to limit the hop length of the possible path
        :param topology: topology graph
        :param source: source node
        :param destination: destination node
        :param original_length: length of the shortest hop path from source to destination
        :return: list of nexthops
        """
        possible_nexthops = []
        checked_nexthops = set()
        my_out_edges = topology.edges(nbunch=source, keys=True)
        for _, neighbor, edge_index in my_out_edges:
            if (neighbor, edge_index) in checked_nexthops:
                continue
            checked_nexthops.add((neighbor, edge_index))

            try:
                if topology is self._forward_ordering:
                    removed_tuple = tuple(sorted(self._forward_removed_nodes))
                    length = self._forward_path_lengths[neighbor][destination][removed_tuple]
                else:
                    removed_tuple = tuple(sorted(self._reverse_removed_nodes))
                    length = self._reverse_path_lengths[neighbor][destination][removed_tuple]
                # length = shortest_path_length(topology, neighbor, destination)
            except KeyError:
                continue
            if length + 1 <= int(original_length * self._length_cutoff):
                possible_nexthops.append(GraphPathElement(from_=source, to_=neighbor, index=edge_index))
        return possible_nexthops

    def _check_change_direction_path_possible(
                self, forward_graph: networkx.MultiDiGraph, reverse_graph: networkx.MultiDiGraph,
                source: str, destination: str, original_length: int) -> bool:
        for _, node_to in dfs_edges(forward_graph, source):
            if node_to == destination:
                continue
            try:
                forward_removed_tuple = tuple(sorted(self._forward_removed_nodes))
                reverse_removed_tuple = tuple(sorted(self._reverse_removed_nodes))
                if forward_graph is self._forward_ordering:
                    depth = self._forward_path_lengths[source][node_to][forward_removed_tuple]
                    length = self._reverse_path_lengths[node_to][destination][reverse_removed_tuple]
                else:
                    depth = self._reverse_path_lengths[source][node_to][reverse_removed_tuple]
                    length = self._forward_path_lengths[node_to][destination][forward_removed_tuple]
                # depth = shortest_path_length(forward_graph, source, node_to)
                # length = shortest_path_length(reverse_graph, node_to, destination)
            except KeyError:
                continue
            if length + depth + 1 <= int(original_length * self._length_cutoff):
                return True

    def _find_nexthops_with_change_direction(
                self, forward_graph: networkx.MultiDiGraph, reverse_graph: networkx.MultiDiGraph,
                source: str, destination: str, original_length: int) -> list[GraphPathElement]:
        """
        Search for possible nexthops by allowing to change the direction once
        Use dfs search in forward_graph looking for a node that has a simple path in reverse_graph
            If there is, here is our nexthop.

        :param forward_graph: initial graph to look through
        :param reverse_graph: graph to look through after changing direction
        :param source: source node
        :param destination: destination node
        :return: list of nexthops
        """
        possible_nexthops = []
        checked_nexthops = set()
        with self._hide_nodes(reverse_graph, [source]):
            my_out_edges = forward_graph.edges(nbunch=source, keys=True)
            for _, neighbor, edge_index in my_out_edges:
                if (neighbor, edge_index) in checked_nexthops:
                    continue
                checked_nexthops.add((neighbor, edge_index))

                try:
                    if reverse_graph is self._forward_ordering:
                        removed_tuple = tuple(sorted(self._forward_removed_nodes))
                        length = self._forward_path_lengths[neighbor][destination][removed_tuple]
                    else:
                        removed_tuple = tuple(sorted(self._reverse_removed_nodes))
                        length = self._reverse_path_lengths[neighbor][destination][removed_tuple]
                    # length = shortest_path_length(reverse_graph, neighbor, destination)
                except KeyError:
                    # if there is no easy path, check change direction
                    if self._check_change_direction_path_possible(
                                forward_graph, reverse_graph, neighbor, destination, original_length):
                        possible_nexthops.append(GraphPathElement(from_=source, to_=neighbor, index=edge_index))
                else:
                    # if easy path exists, check its length and save it
                    if length + 1 <= int(original_length * self._length_cutoff):
                        possible_nexthops.append(GraphPathElement(from_=source, to_=neighbor, index=edge_index))

        return possible_nexthops

    def _calculate_from_previous(self, topology: networkx.MultiDiGraph, source: str,
                                 previous: Optional[str], destination: str, start_node: str) -> list[GraphPathElement]:
        # in our orderings only paths that exist are the ones that go from:
        #   lower dfs_number to higher for forward ordering,
        #   higher dfs_number to lower for reverse ordering
        # so we must use the correct graph depending on source and destination
        max_len = topology.number_of_nodes() + 1
        try:
            if (self._forward_ordering.nodes[source]['dfs_number'] >
                    self._forward_ordering.nodes[previous]['dfs_number']):
                forward_graph = self._forward_ordering
                reverse_graph = self._reverse_ordering
            else:
                forward_graph = self._reverse_ordering
                reverse_graph = self._forward_ordering
        except KeyError:
            # source or destination was removed from topology
            raise networkx.NodeNotFound

        with self._hide_nodes(forward_graph, [previous, start_node]):
            with self._hide_nodes(reverse_graph, [previous, start_node]):
                try:
                    if forward_graph is self._forward_ordering:
                        removed_tuple = tuple(sorted(self._forward_removed_nodes))
                        original_length = self._forward_path_lengths[source][destination][removed_tuple]
                    else:
                        removed_tuple = tuple(sorted(self._reverse_removed_nodes))
                        original_length = self._reverse_path_lengths[source][destination][removed_tuple]
                    # original_length = shortest_path_length(forward_graph, source, destination)
                    return self._find_possible_nexthops(forward_graph, source, destination, original_length)
                except KeyError:
                    pass

                # now check a simple path in the opposite direction, maybe this node is where we change it
                try:
                    if reverse_graph is self._forward_ordering:
                        removed_tuple = tuple(sorted(self._forward_removed_nodes))
                        original_length = self._forward_path_lengths[source][destination][removed_tuple]
                    else:
                        removed_tuple = tuple(sorted(self._reverse_removed_nodes))
                        original_length = self._reverse_path_lengths[source][destination][removed_tuple]
                    # original_length = shortest_path_length(reverse_graph, source, destination)
                    return self._find_possible_nexthops(reverse_graph, source, destination, original_length)
                except KeyError:
                    pass

                # if we are here there is no simple path from source to destination
                # now we look for a node that has a simple path
                # and is reachable from current source by going either only forward or only reverse

                result = self._find_nexthops_with_change_direction(
                        forward_graph, reverse_graph, source, destination, max_len)
                if not result:
                    raise networkx.NetworkXNoPath
                return result

    def calculate(self, topology: networkx.MultiDiGraph, source: str,
                  previous: Optional[str], destination: str, start_node: str) -> list[GraphPathElement]:
        if previous is not None:
            return self._calculate_from_previous(topology, source, previous, destination, start_node)
        max_len = topology.number_of_nodes() + 1

        # previous is none so this is the first node that sees the current flow
        # here we can provide more nexthop options than usual

        possible_nexthops = []

        try:
            original_length_forward = self._forward_path_lengths[source][destination][tuple()]
            # original_length_forward = shortest_path_length(self._forward_ordering, source, destination)
            possible_nexthops.extend(
                    self._find_possible_nexthops(self._forward_ordering, source, destination, original_length_forward)
            )
        except KeyError:
            original_length_forward = max_len

        try:
            original_length_reverse = self._reverse_path_lengths[source][destination][tuple()]
            # original_length_reverse = shortest_path_length(self._reverse_ordering, source, destination)
            possible_nexthops.extend(
                    self._find_possible_nexthops(self._reverse_ordering, source, destination, original_length_reverse)
            )
        except KeyError:
            original_length_reverse = max_len

        original_length = min(original_length_reverse, original_length_forward)

        possible_nexthops.extend(
                self._find_nexthops_with_change_direction(
                        self._forward_ordering, self._reverse_ordering, source, destination, original_length
                )
        )

        possible_nexthops.extend(
                self._find_nexthops_with_change_direction(
                        self._reverse_ordering, self._forward_ordering, source, destination, original_length
                )
        )

        if self._store_nexthops:
            self._flow_set_nexthops.append(len(possible_nexthops))
        if not possible_nexthops:
            raise networkx.NetworkXNoPath
        return possible_nexthops

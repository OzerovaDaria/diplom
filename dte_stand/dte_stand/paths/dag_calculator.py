import networkx
import math
from networkx.algorithms.shortest_paths import shortest_path_length
import networkx.algorithms.simple_paths
from networkx.algorithms.traversal import dfs_tree, bfs_edges, dfs_edges
from networkx.algorithms.dag import dag_longest_path, topological_sort
from typing import Generator, Tuple, List

from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement


class DAGCalculator(BasePathCalculator):
    def __init__(self, length_cutoff_fraction=2.0):
        super().__init__()
        self._forward_ordering = networkx.MultiDiGraph()
        self._reverse_ordering = networkx.MultiDiGraph()
        # how much longer the found paths can be compared to the shortest hop path
        self._length_cutoff = length_cutoff_fraction

    def _get_topology_edges_between_nodes(
                self, topology: networkx.MultiDiGraph,
                node1: str, node2: str) -> Generator[Tuple[int, dict], None, None]:
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

    def _find_possible_nexthops(self, topology: networkx.MultiDiGraph,
                                source: str, destination: str, original_length: int) -> List[List[GraphPathElement]]:
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
            if neighbor in checked_nexthops:
                continue
            checked_nexthops.add(neighbor)

            try:
                length = shortest_path_length(topology, neighbor, destination)
            except networkx.NetworkXNoPath:
                continue
            if length <= int(math.ceil(original_length * self._length_cutoff)):
                possible_nexthops.append([GraphPathElement(from_=source, to_=neighbor, index=edge_index)])
        return possible_nexthops

    def _check_change_direction_path_possible(
                self, forward_graph: networkx.MultiDiGraph, reverse_graph: networkx.MultiDiGraph,
                source: str, destination: str) -> bool:
        for _, node_to in dfs_edges(forward_graph, source):
            if networkx.has_path(reverse_graph, node_to, destination):
                return True

    def _find_nexthops_with_change_direction(
                self, forward_graph: networkx.MultiDiGraph, reverse_graph: networkx.MultiDiGraph,
                source: str, destination: str) -> List[List[GraphPathElement]]:
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
        print
        possible_nexthops = []
        checked_nexthops = set()
        my_out_edges = forward_graph.edges(nbunch=source, keys=True)
        for _, neighbor, edge_index in my_out_edges:
            if neighbor in checked_nexthops:
                continue
            checked_nexthops.add(neighbor)

            if (networkx.has_path(reverse_graph, neighbor, destination) or
                    self._check_change_direction_path_possible(forward_graph, reverse_graph, neighbor, destination)):
                possible_nexthops.append([GraphPathElement(from_=source, to_=neighbor, index=edge_index)])
        return possible_nexthops

    def calculate(self, topology: networkx.MultiDiGraph, source: str, destination: str) -> List[List[GraphPathElement]]:
        # in our orderings only paths that exist are the ones that go from:
        #   lower dfs_number to higher for forward ordering,
        #   higher dfs_number to lower for reverse ordering
        # so we must use the correct graph depending on source and destination
        #print("CALCULATE", source,  destination)
        #print(topology.nodes(data=False))
        #print(topology.edges())

        try:
            #print("FORWARD ORDERING NODES")
            #print(self._forward_ordering.nodes)
            #print(self._forward_ordering.nodes[source])
            if (self._forward_ordering.nodes[source]['dfs_number'] <
                    self._forward_ordering.nodes[destination]['dfs_number']):
                graph_to_use = self._forward_ordering
            else:
                graph_to_use = self._reverse_ordering
        except KeyError:
            # source or destination was removed from topology
            raise networkx.NodeNotFound

        try:
            original_length = shortest_path_length(graph_to_use, source, destination)
            return self._find_possible_nexthops(graph_to_use, source, destination, original_length)
        except networkx.NetworkXNoPath:
            pass

        # if we are here there is no simple path from source to destination
        # now we look for a node that has a simple path
        # and is reachable from current source by going either only forward or only reverse

        result = (
                self._find_nexthops_with_change_direction(
                        self._forward_ordering, self._reverse_ordering, source, destination)
                or
                self._find_nexthops_with_change_direction(
                        self._reverse_ordering, self._forward_ordering, source, destination)
        )
        if not result:
            raise networkx.NetworkXNoPath
        return result

import networkx
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.data_structures import HashWeights, Flow

import logging
LOG = logging.getLogger(__name__)


class DummyAlgorithm(BaseAlgorithm):
    def step(self, topology: networkx.MultiDiGraph, flows: list[Flow], iteration_num: int) -> HashWeights:
        LOG.debug('Running dummy algorithm')
        hash_weights = HashWeights()
        topo_nodes = topology.nodes
        for start_node in topo_nodes:
            for end_node in topo_nodes:
                if start_node == end_node:
                    continue
                try:
                    node_edges = list(topology.edges(nbunch=start_node, keys=True))
                except KeyError:
                    # node was removed from topology
                    continue
                for edge in node_edges:
                    edge_start, edge_end, edge_index = edge
                    hash_weights.put(edge_start, end_node, edge_end, edge_index, 1)
        return hash_weights


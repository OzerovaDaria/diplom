import networkx
from networkx.algorithms.shortest_paths import shortest_path_length
import networkx.algorithms.simple_paths
from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement
from typing import List

class DummyPathCalculator(BasePathCalculator):
    """
    Calculator that returns only shortest paths by hops
    This ensures there will be no cycles but often produces only one path
    """
    def calculate(self, topology: networkx.MultiDiGraph, source: str, destination: str) -> List[List[GraphPathElement]]:
        length = shortest_path_length(topology, source, destination)
        nx_paths = all_simple_edge_paths(topology, source, destination, cutoff=length)
        return [[GraphPathElement(from_=s, to_=d, index=i) for s, d, i in path] for path in nx_paths]


import abc
import networkx
from networkx.exception import NodeNotFound, NetworkXNoPath
from typing import Generator, Optional, Any, List
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.paths.base import BasePathCalculator
from dte_stand.data_structures import GraphPathElement, Bucket


import logging
LOG = logging.getLogger(__name__)


class PathNotFoundError(Exception):
    def __init__(self, current_node: str, flow: Flow):
        message = f'Path not found at node {current_node} for flow {flow} from {flow.start} to {flow.end}'
        super().__init__(message)


class BaseHashFunction(metaclass=abc.ABCMeta):
    def __init__(self, path_calculator: BasePathCalculator, default_weight=1, debug_check_cycles=0):
        self.path_calculator = path_calculator
        self._default_weight = default_weight
        self._check_cycles = debug_check_cycles

    @abc.abstractmethod
    def _choose_nexthop(self, buckets: List[Bucket], flow_id: str) -> Optional[GraphPathElement]:
        ...

    def _flow_path(self, topology: networkx.MultiDiGraph,
                   flow: Flow, hash_weights: HashWeights, current_node,
                   depth: Optional[int] = None) -> List[GraphPathElement]:
        if (current_node and current_node == flow.end) or (depth is not None and depth <= 0):
            return []
        #print(topology.nodes(data=False))
        #print(hash_weights.weights)
        try:
            all_paths = self.path_calculator.calculate(topology, current_node, flow.end)
        except NetworkXNoPath as e:
            raise PathNotFoundError(current_node, flow) from e
        '''
        except NodeNotFound:
            LOG.info(f'One of the nodes ({current_node} or {flow.end}) was removed from topology.'
                     f'Flow ({flow}) is dropped')
            raise
        '''
        #print("PATHS", all_paths)
        try:
            nexthops = [path[0] for path in all_paths]
        except IndexError:
            raise PathNotFoundError(current_node, flow)

        # there might be a case when no bucket exists for a nexthop
        # it happens because hash weights come from previous iteration
        # and on previous iteration topology might have been different so the needed edge is not there
        # for such cases we create a bucket manually and set its weight to default weight
        # it probably (?) makes sense to set this weight as high as possible
        # because the edge is "new" and has no traffic
        all_bucket_edge_dict = {(bucket.edge.from_, bucket.edge.to_, bucket.edge.index): bucket
                                for bucket in hash_weights.get_bucket_list(current_node, flow.end)}
        available_buckets: list[Bucket] = []
        for nexthop in nexthops:
            nexthop_tuple = (nexthop.from_, nexthop.to_, nexthop.index)
            if nexthop_tuple in all_bucket_edge_dict:
                available_buckets.append(all_bucket_edge_dict[nexthop_tuple])
            else:
                available_buckets.append(Bucket(edge=nexthop, weight=self._default_weight))

        chosen_nexthop = self._choose_nexthop(available_buckets, flow.flow_id)
        if not chosen_nexthop:
            raise PathNotFoundError(current_node, flow)

        flow_path = [chosen_nexthop]
        depth = depth - 1 if depth is not None else None
        flow_path.extend(self._flow_path(topology, flow, hash_weights, chosen_nexthop.to_, depth=depth))
        return flow_path

    def _check_cycle(self, path: List[GraphPathElement]):
        only_nodes = [elem.from_ for elem in path]
        only_nodes.append(path[-1].to_)
        only_nodes_set = set(only_nodes)

        if len(only_nodes_set) != len(only_nodes):
            LOG.warning(f'Cycle detected in path: {path}')

    def run(self, topology: networkx.MultiDiGraph, flows: List[Flow],
            hash_weights: HashWeights, fl, depth: Optional[int] = None) -> None:
        """
        Main function to run hash

        :param topology: current topology
        :param flows: current list of flows
        :param hash_weights: current hash weights
        :param depth: if None, full path will be found.
            If positive int, path finding will stop after <depth> hops. The rest of the path will not be calculated
                and bandwidths will not change
            If negative int or 0, all paths will be empty
        :return:
        """
        #print("FL = ", fl)
        for _, _, edge_data in topology.edges(data=True):
            edge_data['current_bandwidth'] = 0
        flow_paths = {}
        for flow in flows:
            try:
                flow_path = self._flow_path(topology, flow, hash_weights, flow.start, depth=depth)
                if fl:
                    print("FLOW PATH", flow_path)
            except PathNotFoundError:
                LOG.exception('Failed to find path for flow: ')
                continue
            except NodeNotFound:
                # debug message is cought inside _flow_path
                continue
            if self._check_cycles:
                self._check_cycle(flow_path)
            flow_paths[flow.flow_id] = flow_path
            for element in flow_path:
                #print(flow_path)
                if element.from_ not in topology.nodes() or element.to_ not in topology.nodes():
                    continue
                #print(topology.edges[element.from_, element.to_, element.index]['current_bandwidth'])
                topology.edges[element.from_, element.to_, element.index]['current_bandwidth'] += flow.bandwidth
        return flow_paths

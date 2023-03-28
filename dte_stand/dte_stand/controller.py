import networkx
from dte_stand.data_structures import HashWeights, Flow, InputData
from dte_stand.hash_function.base import BaseHashFunction
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.paths.base import BasePathCalculator
from typing import Optional, Tuple, List
#import matplotlib.pyplot as plt
import cProfile
import pstats
import pymetis
import numpy as np
import copy
import multiprocessing as mp
import dill
import random
from multiprocessing.dummy import Pool
from pstats import SortKey

import logging
LOG = logging.getLogger(__name__)


class ExperimentController:
    def __init__(self, path_to_inputs: str, lsdb_period: int, num_iterations: int,
                 hash_function: BaseHashFunction, algorithm: BaseAlgorithm, path_calculator: BasePathCalculator):
        self.input_data = InputData(path_to_inputs)
        self.period = lsdb_period
        self.num_iterations = num_iterations
        self.hash_function = hash_function
        self.algorithm = algorithm
        self.path_calculator = path_calculator
        self.result_list = []

    def log_result(self, result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        self.result_list.append(result)

    def _get_current_topology_and_time(self, current_time: int) -> Tuple[networkx.MultiDiGraph, int]:
        # get topology and time when topology last changed
        #print(current_time, type(current_time))
        current_topology, change_time = self.input_data.topology.get(current_time + self.period)

        # if topology changed between (current_time, current_time+period),
        # then current time is actually the time when it changed,
        # because our algorithm was run because of the change, not because a period has passed
        if (change_time is not None) and (current_time < change_time < current_time + self.period):
            return current_topology, change_time

        # otherwise, no changes to topology in the experiment yet, or the change was too long ago
        return current_topology, current_time + self.period

    def _calculate_current_bandwidth(self, topology: networkx.MultiDiGraph, flows: List[Flow],
                                     hash_weights: HashWeights, fl) -> None:
        if hash_weights == '0':
            # first iteration
            return
        return self.hash_function.run(topology, list(flows), hash_weights, fl)

    def _calculate_phi(self, topology: networkx.MultiDiGraph) -> float:
        number_of_edges = topology.number_of_edges()
        average_load: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            average_load += float(edge_data['current_bandwidth']) / edge_data['bandwidth']
        average_load /= number_of_edges

        deviation: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            deviation += pow(float(edge_data['current_bandwidth']) / edge_data['bandwidth'] - average_load, 2)
        return deviation / number_of_edges

    def generate_subgraphs(self, topology, num_of_subgraphs):
        my_out_edges = topology.edges(nbunch='2', keys=True)
        for a, neighbor, edge_index in my_out_edges:
            print(a, neighbor, edge_index)
        adjacency_list = []
        for i in topology.nodes():
            if np.fromiter(topology.neighbors(i), int).size != 0:
                adjacency_list.append(np.fromiter(topology.neighbors(i), int))
        #print("ADJENCY matrix", adjacency_list, len(adjacency_list), type(adjacency_list))
        n_cuts, membership = pymetis.part_graph(num_of_subgraphs, adjacency=adjacency_list)
        subgraphs = []
        nodes = []
        routers = [{} for i in range(num_of_subgraphs)]
        hypergraph = networkx.MultiDiGraph()
        for i in range(num_of_subgraphs):
            hypergraph.add_node(str(i))
            current_topology: networkx.MultiDiGraph = copy.deepcopy(topology)
            #print(i, np.argwhere(np.array(membership) == i).ravel())
            nodes.append(np.argwhere(np.array(membership) == i).ravel())
            nodes[i] = [str(x) for x in nodes[i]]
            current_nodes = topology.nodes
            for node in current_nodes:
                if node not in nodes[i]:
                    current_topology.remove_node(node)
            subgraphs.append(current_topology)
            #print("my topo", subgraphs[i].nodes, subgraphs[i].edges())
        for i in range(num_of_subgraphs):
            for j in nodes[i]:
                #print(j, end='')
                for link in topology.in_edges(str(j)):
                    if link[0] not in nodes[i]:
                        for k in range(len(nodes)):
                            if link[0] in nodes[k]: #and not hypergraph.has_edge(str(i), str(k)):
                                if hypergraph.has_edge(str(i), str(k)):
                                    hypergraph.add_edge(str(i), str(k), keys=1, weight=random.randint(1, 7))
                                else:
                                    hypergraph.add_edge(str(i), str(k), keys=0, weight=random.randint(1, 7))
                '''
                for link in topology.out_edges(str(j)):
                    if link[1] not in nodes[i]:
                        for k in range(len(nodes)):
                            if link[1] in nodes[k]:
                                hypergraph.add_edge(k, i)
                '''
        #print("HYPERGRAPH")
        #print(hypergraph.nodes, hypergraph.edges(data=True))
        routers[0] = {'start': '0', 'end': '4'}
        routers[1] = {'start': '8', 'end': '14'}

        return subgraphs, hypergraph, routers

    def return_callback(self, result):
        print(f'Callback received: {result}', flush=True)

    def balance_hypergraph(self, subgraphs, hypergraph, current_flows, routers):
        hypergraph_flows = []
        subgraph_flows = [[] for x in range(len(subgraphs))]
        start, end = 0, 0
        #print("CURRENT FLOWS", current_flows)
        for flow in current_flows:
            for i in range(len(subgraphs)):
                if flow.start in subgraphs[i]:
                    start = str(i)
                if flow.end in subgraphs[i]:
                    end = str(i)
            if start == end:
                subgraph_flows[int(start)].append(flow)
            else:
                hypergraph_flows.append(Flow(start=start, end=end, all_bandwidth=flow.all_bandwidth,
                              start_time=flow.start_time,
                              end_time=flow.end_time, bandwidth=flow.bandwidth, flow_id=flow.flow_id))
        #print("HYPERGRAPH FLOWS")
        #print(hypergraph_flows)
        print("SUBGRAPH FLOWS")
        print("SUB 0")
        print(subgraph_flows[0])
        print("SUB 1")
        print(subgraph_flows[1])
        hypergraph_hw = self.get_HashWeights(hypergraph)
        self.path_calculator.prepare_iteration(hypergraph)
        flow_paths = self._calculate_current_bandwidth(hypergraph, hypergraph_flows, hypergraph_hw, True)
        print(flow_paths)
        fl = 0
        for flow in current_flows:
            fl = 0
            for i in range(len(subgraph_flows)):
                if flow in subgraph_flows[i]:
                    fl = 1
            if fl == 1:
                continue
            for path_elem in flow_paths[flow.flow_id]:
                print("PATH ELEM", path_elem)
                for i in range(len(subgraphs)):
                    if path_elem.from_ == str(i):
                        if flow.start != routers[i]['end']:
                            print("from", i, flow.start, routers[i]['end'], type(flow.start), type(routers[i]['end']))
                            subgraph_flows[i].append(Flow(start=flow.start, end=routers[i]['end'], all_bandwidth=flow.all_bandwidth,
                                  start_time=flow.start_time,
                                  end_time=flow.end_time, bandwidth=flow.bandwidth, flow_id=flow.flow_id))
                    if path_elem.to_ == str(i):
                        if flow.end != routers[i]['start']:
                            print("to", i, routers[i]['start'], flow.end)
                            subgraph_flows[i].append(Flow(start=routers[i]['start'], end=flow.end, all_bandwidth=flow.all_bandwidth,
                                  start_time=flow.start_time,
                                  end_time=flow.end_time, bandwidth=flow.bandwidth, flow_id=flow.flow_id))
        print("SUBGRAPH FLOWS NEW")
        print("SUB 0 NEW")
        print(subgraph_flows[0])
        print("SUB 1 NEW")
        print(subgraph_flows[1])
        return subgraph_flows, flow_paths

    def get_HashWeights(self, topology):
        hash_weights = HashWeights()
        topo_nodes = topology.nodes()
        for start_node in topo_nodes:
            for end_node in topo_nodes:
                if start_node == end_node:
                    continue
                try:
                    node_edges = list(topology.edges(nbunch=start_node, keys=True))
                except KeyError:
                    print("node was removed from topology")
                    continue
                for edge in node_edges:
                    edge_start, edge_end, edge_index = edge
                    edge_weight = (topology.get_edge_data(edge_start, edge_end, edge_index)[
                        "weight"])
                    hash_weights.put(edge_start, end_node, edge_end, edge_index, edge_weight)
        return hash_weights



    def run(self) -> float:
        #print("Number of processors", mp.cpu_count())
        hash_weights: Optional[HashWeights] = None
        current_time = -self.period
        merged_dict = {}
        num_of_subgraphs = 2
        subgraph_hw = []
        for iteration in range(self.num_iterations):
            current_topo, current_time = self._get_current_topology_and_time(current_time)
            #print(current_topo.nodes(), "\n", current_topo.edges()) #data=True
            #print(current_topo.in_edges('0'))
            #print(current_topo.out_edges('0'))
            subgraphs, hypergraph, routers = self.generate_subgraphs(current_topo, num_of_subgraphs)

            print("SUBGRAPHS", subgraphs)
            print("HYPERGRAPH")
            print(hypergraph.nodes, hypergraph.edges)

            LOG.info(f'current time: {current_time}')
            current_flows = self.input_data.flows.get(current_time)
            #print("CURRENT", current_flows)
            flows, flow_paths = self.balance_hypergraph(subgraphs, hypergraph, current_flows, routers)
            #
            '''
            if len(subgraph_hw) > 0:
                for i in range(len(subgraphs)):
                    #print("HASH W", subgraph_hw[i])
                    self.path_calculator.prepare_iteration(subgraphs[i])
                    fl_path = self._calculate_current_bandwidth(subgraphs[i], flows[i], subgraph_hw[i], True)
                    print("BALANCED IN SB", subgraphs[i].nodes)
                    print(subgraphs[i].edges(data=True))

            phi = self._calculate_phi(current_topo)
            LOG.info(f'Iteration: {iteration}, phi: {phi}')
            #pr = cProfile.Profile()
            #pr.enable()

            pool = mp.Pool()
            
            #results = [pool.apply_async(self.algorithm.step, args=(subgraphs[i], current_flows, iteration, i))
            #       for i in range(len(subgraphs))]
            
            for i in range(len(subgraphs)):
                pool.apply_async(self.algorithm.step, args=(subgraphs[i], current_flows, iteration, i),
                                            callback=self.log_result)
            #hash_weights, phi_dict = self.algorithm.step(current_topo, current_flows, iteration)
            pool.close()
            pool.join()
            #output = [p.get(0) for p in self.result_list]
            print("OUTPUT", self.result_list)

            for i in range(len(subgraphs)):
                with open("hw-" + str(i) + "-pickle", 'rb') as file:
                    hw = dill.load(file)
                subgraph_hw.append(hw)
            #for i in subgraph_hw:
            #    print(i.weights)

            #print("RES", results)
            #pr.disable()
            #sortby = 'cumulative'
            #ps = pstats.Stats(pr).sort_stats(sortby)
            #ps.print_stats(150)

            #merged_dict = self.update_phi_dct(merged_dict, phi_dict, iteration)

            #print("MERGED DICT", merged_dict)
            
        #self._calculate_current_bandwidth(current_topo, current_flows, hash_weights, True) #!!!

        phi = self._calculate_phi(current_topo)
        #self.phi_graph(merged_dict)
        LOG.info(f'phi after experiment: {phi}')
        return phi
        '''

    def update_phi_dct(self, merged_dict, new_dict, iteration):
        for key, value in new_dict.items():
            merged_dict[key + iteration * len(new_dict)] = value
        return merged_dict
    '''
    def phi_graph(self, phi_dct=None):
        x = list(phi_dct.keys())
        y = list(phi_dct.values())
        # plt.plot(x, y)
        # plt.xlabel('number of episode')
        # plt.ylabel('phi')
        # plt.title('PHI VALUES')
        # plt.legend()
        # plt.savefig("Phi.png")
        #plt.figure(figsize=(15, 8))
        figure, ax = plt.subplots(figsize=(15, 8))
        ax.plot(x, y, color='blue', marker='o', linestyle='dashed', linewidth=0.8, markersize=2, label='phi')
        # ax.plot(x, y, label='phi')
        ax.set_xlabel('iteration', fontsize=10)
        ax.set_ylabel('phi value', fontsize=10)
        ax.legend()
        plt.savefig('phi/final_plot.png')
    '''
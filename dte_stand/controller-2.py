import networkx
import os
import random
import copy
import numpy as np
import pymetis
from dte_stand.data_structures import HashWeights, Flow, InputData
from dte_stand.hash_function.base import BaseHashFunction
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.paths.base import BasePathCalculator
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.history import HistoryTracker
from dte_stand.config import Config
from typing import Optional, Callable

import logging
LOG = logging.getLogger(__name__)

import multiprocessing as mp
import time
import dill

from dte_stand.config import Config
import importlib

def dynamic_import_function(object_path):
    module, object, function_name = object_path.rsplit('.', 2)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return getattr(obj_class, function_name)


def dynamic_import(object_path: str, **module_kwargs):
    module, object = object_path.rsplit('.', 1)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return obj_class(**module_kwargs)

def get_submarl(experiment_folder="data_examples/huawei"):
    Config.load_config(experiment_folder)
    config = Config.config()

    phi_func = dynamic_import_function(config.phi)
    path_calculator = dynamic_import(config.path_calculator)
    hash_function = dynamic_import(config.hash_function, path_calculator=path_calculator,
                                   debug_check_cycles=config.debug_check_cycles)
    algo = dynamic_import(config.algorithm, hash_function=hash_function, phi_func=phi_func, experiment_dir="data_examples/huawei/parallel",
                          model_dir=None)
    return algo, path_calculator, hash_function

def foo_pool(x, current_topo, current_flows, iteration):
    #time.sleep(2)
    alg, path_calc, hash_func = get_submarl()
    #print(x, current_topo.nodes, len(current_flows), iteration)
    print("GOT IN THREAD")
    path_calc.prepare_iteration(current_topo)
    print("PREPARED ITER IN THREAD")
    #calculate_current_bandwidth(current_topo, current_flows, hash_weights)
    print("THREAD", current_topo.nodes,"\n\n", current_flows, "\n\n")
    hash_weights = alg.step(current_topo, current_flows, iteration)
    with open("hw-" + str(x) + "-pickle", "wb") as f:
            dill.dump(hash_weights, f)
    print(x, "HASH_WEIGHTS", hash_weights)
    #PhiCalculator.end_iteration_and_plot_graph()
    #PhiCalculator.plot_full(all_iterations=False)
    #hash_func.end_iteration()
    #return x*x

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def apply_async_with_callback(current_topo, current_flows, iteration):
    #for i in range(len(current_topo)):
    #    print("ASYNC TOPO", current_topo[i].nodes, current_topo[i].edges)
    #    print("ASYNC FL", current_flows[i])
    print("APPLY ASYNC")
    pool = mp.Pool()
    for i in range(len(current_topo)):
        pool.apply_async(foo_pool, args = (i, current_topo[i], current_flows[i], iteration, ))
    pool.close()
    pool.join()
    #print(result_list)

class ExperimentController:
    def __init__(self, path_to_inputs: str, lsdb_period: int, num_iterations: int,
                 hash_function: BaseHashFunction, algorithm: BaseAlgorithm, path_calculator: BasePathCalculator,
                 phi_func: Callable, experiment_dir: str):
        self.input_data = InputData(path_to_inputs)
        self.period = lsdb_period
        self.num_iterations = num_iterations
        self.hash_function = hash_function
        self.algorithm = algorithm
        self.path_calculator = path_calculator
        self.phi = phi_func
        self.experiment_dir = experiment_dir

    def _get_current_topology_and_time(self, current_time: int) -> tuple[networkx.MultiDiGraph, int]:
        # get topology and time when topology last changed
        current_topology, change_time = self.input_data.topology.get(current_time + self.period)

        # if topology changed between (current_time, current_time+period),
        # then current time is actually the time when it changed,
        # because our algorithm was run because of the change, not because a period has passed
        if (change_time is not None) and (current_time < change_time < current_time + self.period):
            return current_topology, change_time

        # otherwise, no changes to topology in the experiment yet, or the change was too long ago
        return current_topology, current_time + self.period

    def _calculate_current_bandwidth(self, topology: networkx.MultiDiGraph, flows: list[Flow],
                                     hash_weights: HashWeights) -> None:
        if hash_weights is None:
            # first iteration
            return
        return self.hash_function.run(topology, list(flows), hash_weights)

    def _end_iteration(self):
        PhiCalculator.end_iteration_and_plot_graph()
        PhiCalculator.plot_full(all_iterations=False)
        self.hash_function.end_iteration()

    def generate_subgraphs(self, topology, num_of_subgraphs):
        #print("TOPO", topology.edges(data=True))
        my_out_edges = topology.edges(nbunch='2', keys=True)
        #for a, neighbor, edge_index in my_out_edges:
        #    print(a, neighbor, edge_index)
        #print("TOPO NODES", topology.nodes())
        mapping = {}
        for i in topology.nodes():
            mapping[i] = str(int(i) - 1)
            #print(i, type(i), int(i), str(int(i) - 1))
        topology = networkx.relabel_nodes(topology, mapping)
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
        
        hyp = networkx.MultiDiGraph()
        
        for i in range(num_of_subgraphs):
            mapping = {}
            weights_dct = {}
            hypergraph.add_node(str(i))
            current_topology: networkx.MultiDiGraph = copy.deepcopy(topology)
            #print(i, np.argwhere(np.array(membership) == i).ravel())
            nodes.append(np.argwhere(np.array(membership) == i).ravel())
            nodes[i] = [str(x) for x in nodes[i]]
            current_nodes = topology.nodes
            for node in current_nodes:
                if node not in nodes[i]:
                    current_topology.remove_node(node)
            for i in current_topology.nodes():
                mapping[i] = str(int(i) + 1)
            #for j in current_topology.edges:
            #    weights_dct[j] = { "weight" : random.randint(1, 7)}
            #networkx.set_edge_attributes(current_topology, weights_dct)
            subgraphs.append(networkx.relabel_nodes(current_topology, mapping))
            #print("my topo", subgraphs[i].nodes, subgraphs[i].edges())
        for i in range(num_of_subgraphs):
            for j in nodes[i]:
                #print(j, end='')
                for link in topology.in_edges(str(j)):
                    #print("LINK", link[0], link[1])
                    if link[0] not in nodes[i]:
                        for k in range(len(nodes)):
                            if link[0] in nodes[k]:
                                #hyp.add_node(link[0])
                                #hyp.add_node(link[1])
                                #hyp.add_edge(*link)
                                #print("LINK HW", link) #and not hypergraph.has_edge(str(i), str(k)):
                                if  hypergraph.has_edge(str(i), str(k)):
                                    hypergraph.add_edge(str(i), str(k), keys=1, id='0', bandwidth=40000000, current_bandwidth=0, weight=random.randint(1, 7), 
                                                                                        start=link[0], end=link[1])
                                    hyp.add_edge(link[0], link[1], keys=1, id='0', bandwidth=40000000, weight=random.randint(1, 7))
                                else:
                                    hypergraph.add_edge(str(i), str(k), keys=0, id='0', bandwidth=40000000, current_bandwidth=0, weight=random.randint(1, 7),
                                                                                        start=link[0], end=link[1])
                                    hyp.add_edge(link[0], link[1], keys=0, id='0', bandwidth=40000000, weight=random.randint(1, 7))
                '''
                for link in topology.out_edges(str(j)):
                    if link[1] not in nodes[i]:
                        for k in range(len(nodes)):
                            if link[1] in nodes[k]:
                                hypergraph.add_edge(k, i)
                '''
        #print("HYPERGRAPH")
        #print(hypergraph.nodes, hypergraph.edges(data=True))
        #print("HYP")
        #print(hyp.nodes, hyp.edges(data=True))
        #for i in range(len(subgraphs)):
        #    print(subgraphs[i].edges(data=True))
        routers[0] = {'start': '1', 'end': '5'}
        routers[1] = {'start': '9', 'end': '15'}

        #for i in range(len(subgraphs)):
        #    print("SUB", i, subgraphs[i].nodes)
        return subgraphs, hypergraph, routers, hyp

    def balance_hypergraph(self, subgraphs, hypergraph, hyp, current_flows, routers, iteration):
        hypergraph_flows, hyp_fl = [], []
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
                hyp_fl.append(Flow(start=flow.start, end=flow.end, all_bandwidth=flow.all_bandwidth,
                              start_time=flow.start_time,
                              end_time=flow.end_time, bandwidth=flow.bandwidth, flow_id=flow.flow_id))
        
        #print("HYPERGRAPH FLOWS")
        #print(hypergraph_flows)
        #print("SUBGRAPH FLOWS")
        #print("SUB 0", subgraph_flows[0], "\n\n\n")
        #print(subgraph_flows[0])
        #print("SUB 1", subgraph_flows[1], "\n\n\n")
        #print(subgraph_flows[1])
        
        hypergraph_hw = self.get_HashWeights(hypergraph)
        hyp_hw = self.get_HashWeights(hyp)
        #print("RANDOM HW", hyp_hw._weights, "\n\n\n")
        print("RANDOM HW", hypergraph_hw._weights)
        self.path_calculator.prepare_iteration(hypergraph)
        
        hypergraph_hw = self.algorithm.step(hypergraph, hypergraph_flows, iteration_num=iteration, save_model=True)
        
        #print("HW", hypergraph_hw._weights)
        #print(type(hypergraph_hw._weights.keys()))
        hyp_keys = list(hypergraph_hw._weights.keys())
        '''
        for key in hyp_keys:
            #for bucket in hypergraph_hw._weights[key]:
                #bucket.weight = int(bucket.weight)
                #print("BUCK", bucket)
            #print("HYP KEYS", key, key[0], key[1])
            if key[0] == 'graph_data' or key[1] == 'graph_data':
                #print("COME IN")
                #hyp_hw._weights.pop(key)
                 hypergraph_hw._weights.pop(key)
        '''
        hypergraph_hw._weights.pop(('0', 'graph_data'), {})
        hypergraph_hw._weights.pop(('1', 'graph_data'), {})
        
        for k in hyp_keys:
            for kk in hypergraph_hw._weights[k]:
                buck_lst = [hypergraph_hw._weights[k][kk]]
                for b in buck_lst:
                    b.weight = int(b.weight)
        #self._calculate_current_bandwidth(hypergraph, hypergraph_flows, hypergraph_hw)
        #hypergraph_hw = self.get_HashWeights(hypergraph)
        #print("HYPERGRAPH FLOWS", hypergraph_flows, "\n\n\n", "HYPERRAPH HW", hypergraph_hw, "\n\n\n")
        #print("HYP HW", hypergraph_hw._weights, "\n\n\n")
        
        #self._end_iteration()
        self.path_calculator.prepare_iteration(hypergraph)
        print("BEFORE FLOW PATH")
        print("TOPO", hypergraph.nodes, "\n", hypergraph.edges, "\n\n\n")
        print("FLOWS", hypergraph_flows, "\n\n\n")
        print("HYP HW", hypergraph_hw._weights, "\n\n\n")
        flow_paths = self._calculate_current_bandwidth(hypergraph, hypergraph_flows, hypergraph_hw)
        print("FLOW PATHS", flow_paths)
        #self._end_iteration()
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
        #print("SUBGRAPH FLOWS NEW")
        #print("SUB 0 NEW")
        #print(subgraph_flows[0])
        #print("SUB 1 NEW")
        #print(subgraph_flows[1])
        return subgraph_flows, flow_paths, hypergraph_hw

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
        hash_weights: Optional[HashWeights] = None
        current_time = -self.period
        num_of_subgraphs = 2
        #file = open("iter_phi.csv", "w")
        for iteration in range(self.num_iterations):
            start_time = time.time()

            iteration_path = os.path.join(self.experiment_dir, f'iteration{iteration}')
            HistoryTracker.set_result_folder(iteration_path)
            PhiCalculator.set_plot_folder(iteration_path)

            current_topo, current_time = self._get_current_topology_and_time(current_time)
            #print("TOPO", current_topo.edges)
            LOG.info(f'current time: {current_time}')
            subgraphs, hypergraph, routers, hyp = self.generate_subgraphs(current_topo, num_of_subgraphs)
            #print("SUBGRAPHS", subgraphs)
            #print("HYPERGRAPH")
            #print(hypergraph.nodes, hypergraph.edges)

            current_flows = self.input_data.flows.get(current_time)
            flows, flow_paths, hash_weights = self.balance_hypergraph(subgraphs, hypergraph, hyp, current_flows, routers, iteration)

            print("OUT OF FUNCTION")
            #subgraph_hw = []
            apply_async_with_callback(subgraphs, flows, iteration) #####

            #hash_weights._weights = {}
            for i in range(len(subgraphs)):
                with open("hw-" + str(i) + "-pickle", 'rb') as file:
                    hw = dill.load(file)
                    #subgraph_hw.append(hw)
                    hash_weights._weights.update(hw._weights)
                    #print("SUBGRAPH HW:", hw._weights)

            #for i in subgraph_hw:
            #    print("SUBGRAPH:", i._weights)

            
            self.path_calculator.prepare_iteration(current_topo)
            #print("HASH_WEIGHTS BEFORE BALANCE", hash_weights._weights)
            self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
            phi = self.phi(current_topo)
            
            LOG.info(f'Iteration: {iteration}, phi: {phi}')
            #file.write(str(iteration) + ',' + str(phi) + ',')
            with open('iter_phi_par.csv', 'a') as file:
                file.write(str(iteration) + ',' + str(phi) + ',' + str(len(current_flows)) + ',' + str(time.time() - start_time) + '\n')

            #hash_weights = self.algorithm.step(current_topo, current_flows, iteration_num=iteration, save_model=True)
            #print("HASH_WEIGHTS", hash_weights._weights)
            self._end_iteration()
        self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
        phi = self.phi(current_topo)
        LOG.info(f'phi after experiment: {phi}')
        #file.close()
        return phi


class RandomExperimentController(ExperimentController):
    def __init__(self, path_to_inputs: str, lsdb_period: int, num_iterations: int,
                 hash_function: BaseHashFunction, algorithm: BaseAlgorithm, path_calculator: BasePathCalculator,
                 phi_func: Callable, experiment_dir: str):
        super().__init__(path_to_inputs, lsdb_period, num_iterations, hash_function,
                         algorithm, path_calculator, phi_func, experiment_dir)
        config = Config.config()
        self._plot_period = config.plot_period
        self._time_points = []
        random.seed()

    def _generate_time_points(self):
        self._time_points = list(range(0, 50 * 30000, self.period))
        random.shuffle(self._time_points)

    def run(self):
        hash_weights: Optional[HashWeights] = None
        current_topo, _ = self.input_data.topology.get(0)
        self.path_calculator.prepare_iteration(current_topo)
        for iteration in range(self.num_iterations):
            try:
                current_time = self._time_points.pop()
            except IndexError:
                self._generate_time_points()
                current_time = self._time_points.pop()
            LOG.info(f'current time: {current_time}')
            current_flows = self.input_data.flows.get(current_time)

            self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
            phi = self.phi(current_topo)
            LOG.info(f'Iteration: {iteration}, phi: {phi}')
            save_model = True if iteration == self.num_iterations - 1 else False

            hash_weights = self.algorithm.step(current_topo, current_flows,
                                               iteration_num=iteration, save_model=save_model)

            self.hash_function.end_iteration()
            if iteration > 0 and ((iteration + 1) % self._plot_period == 0):
                PhiCalculator.plot_result()
        self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
        phi = self.phi(current_topo)
        LOG.info(f'phi after experiment: {phi}')
        PhiCalculator.plot_full(all_iterations=False)
        return phi
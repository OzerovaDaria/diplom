import networkx
import os
import random
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
    print(x, current_topo.nodes, len(current_flows), iteration)
    path_calc.prepare_iteration(current_topo)
    #calculate_current_bandwidth(current_topo, current_flows, hash_weights)
    hash_weights = alg.step(current_topo, current_flows, iteration)
    with open("hw-" + str(x) + "-pickle", "wb") as f:
            dill.dump(hash_weights, f)
    print(x, "HASH_WEIGHTS", hash_weights)
    return x*x

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def apply_async_with_callback(current_topo, current_flows, iteration):
    pool = mp.Pool()
    for i in range(10):
        pool.apply_async(foo_pool, args = (i, current_topo, current_flows, iteration, ), callback = log_result)
    pool.close()
    pool.join()
    print(result_list)

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
        self.hash_function.run(topology, list(flows), hash_weights)

    def _end_iteration(self):
        PhiCalculator.end_iteration_and_plot_graph()
        PhiCalculator.plot_full(all_iterations=False)
        self.hash_function.end_iteration()

    def run(self) -> float:
        hash_weights: Optional[HashWeights] = None
        current_time = -self.period
        for iteration in range(self.num_iterations):
            iteration_path = os.path.join(self.experiment_dir, f'iteration{iteration}')
            HistoryTracker.set_result_folder(iteration_path)
            PhiCalculator.set_plot_folder(iteration_path)

            current_topo, current_time = self._get_current_topology_and_time(current_time)
            LOG.info(f'current time: {current_time}')
            current_flows = self.input_data.flows.get(current_time)

            subgraph_hw = []
            apply_async_with_callback(current_topo, current_flows, iteration) #####

            for i in range(10):
                with open("hw-" + str(i) + "-pickle", 'rb') as file:
                    hw = dill.load(file)
                subgraph_hw.append(hw)
            for i in subgraph_hw:
                print("SUBGRAPH:", i._weights)

            self.path_calculator.prepare_iteration(current_topo)
            self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
            phi = self.phi(current_topo)
            LOG.info(f'Iteration: {iteration}, phi: {phi}')

            hash_weights = self.algorithm.step(current_topo, current_flows, iteration_num=iteration, save_model=True)

            self._end_iteration()
        self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
        phi = self.phi(current_topo)
        LOG.info(f'phi after experiment: {phi}')
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

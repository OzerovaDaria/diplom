import networkx
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.algorithm.mate.lib.run_experiment import Runner
import dill
import time

import logging
LOG = logging.getLogger(__name__)

from typing import List

class MateAlgorithm(BaseAlgorithm):
    def step(self, topology: networkx.MultiDiGraph, flows: List[Flow], iteration, i) -> HashWeights:
        #print(topology.nodes, topology.edges)
        #print("MARL FLOWS", flows)
        phi_dict = {}
        time.sleep(2)
        LOG.debug('Running dummy algorithm')
        if iteration == 0:
            hash_weights, phi_dict = Runner(topology, self.hash_function, False).run_experiment(topology, flows, iteration)
        else:
            hash_weights, phi_dict = Runner(topology, self.hash_function, True).run_experiment(topology, flows, iteration)

        with open("hw-" + str(i) + "-pickle", "wb") as f:
            dill.dump(hash_weights, f)

        return phi_dict
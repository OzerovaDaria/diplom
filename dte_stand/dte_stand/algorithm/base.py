import abc
import networkx
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.hash_function.base import BaseHashFunction
from typing import List

class BaseAlgorithm(metaclass=abc.ABCMeta):
    def __init__(self, hash_function: BaseHashFunction):
        self.hash_function = hash_function

    @abc.abstractmethod
    def step(self, topology: networkx.MultiDiGraph, flows: List[Flow]) -> HashWeights:
        """
        Main function for algorithm

        :param topology: current topology. May be used freely
        :param flows: list of current flows. Flow parameters cannot be used by algorithm.
            Flows are passed only for cases when hash function needs to use them to recalculate current bandwidth
        :return: final hash weights
        """
        ...

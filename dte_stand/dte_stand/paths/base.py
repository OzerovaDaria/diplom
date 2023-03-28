import abc
import networkx
from typing import Optional
from dte_stand.data_structures import GraphPathElement
from typing import List


class BasePathCalculator:
    def prepare_iteration(self, topology: networkx.MultiDiGraph) -> None:
        ...

    @abc.abstractmethod
    def calculate(self, topology: networkx.MultiDiGraph, source: str,
                  previous: Optional[str], destination: str) -> List[List[GraphPathElement]]:
        ...

import networkx
from networkx.exception import NodeNotFound
from dte_stand.data_structures import HashWeights, Flow, Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction, PathNotFoundError
from typing import Optional, List
import random
import copy

import logging
LOG = logging.getLogger(__name__)

class HashFunction(BaseHashFunction):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def get_weight(self, bucket):
        return bucket.weight

    def _choose_nexthop(self, buckets: List[Bucket], flow_id: str) -> Optional[GraphPathElement]:
        if buckets:
            max_weight = max(buckets, key=self.get_weight).weight
            if max_weight <= 0:
                return None
            buckets_averaged = copy.deepcopy(buckets)
            for i in buckets_averaged:
                i.weight /= max_weight
            random.seed(flow_id, version=2)
            while buckets_averaged:
                key = random.randint(0, 10000)
                bucket_number = key % len(buckets_averaged)
                bucket_weight = self.get_weight(buckets_averaged[bucket_number])
                if bucket_weight:
                    random.seed()
                    hash_weight = random.random()
                    if hash_weight < bucket_weight:
                        #LOG.debug(f'nexthop: {buckets_averaged[bucket_number]}')
                        #LOG.info(f'nexthop: {buckets_averaged[bucket_number].edge,buckets_averaged[bucket_number].weight}')
                        return buckets_averaged[bucket_number].edge
                del buckets_averaged[bucket_number]
        return None

    def run(self, topology: networkx.MultiDiGraph, flow: Flow,
            hash_weights: HashWeights, depth: Optional[int] = 1) -> None:
        try:
            flow_path = self._flow_path(topology, flow, hash_weights, flow.start, depth=depth)
            return flow_path
        except PathNotFoundError:
            LOG.exception('Failed to find path for flow: ')
        except NodeNotFound:
            LOG.exception('Failed to find path for flow: ')
        if self._check_cycles:
            self._check_cycle(flow_path)
        return None
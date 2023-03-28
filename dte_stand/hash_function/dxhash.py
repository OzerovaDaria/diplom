from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction
import libscrc
import random
import copy

import logging
LOG = logging.getLogger(__name__)


class WeightedDxHashFunction(BaseHashFunction):
    def __init__(self, *args, **kwargs):
        self._flow_to_bucket_map: dict[tuple[str, str], GraphPathElement] = {}
        return super().__init__(*args, **kwargs)

    def get_weight(self, bucket):
        return bucket.weight

    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True) -> Optional[GraphPathElement]:
        if buckets:
            random.seed(flow_id)
            if use_flow_memory and (self._flow_to_bucket_map.get((flow_id, buckets[0].edge.from_))
                                    and self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)] in
                                            list(bucket.edge for bucket in buckets)):
                # LOG.info(f'nexthop: {self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)]}')
                return self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)]
            max_weight = max(buckets, key=self.get_weight).weight
            if max_weight <= 0:
                return None
            buckets_averaged = copy.deepcopy(buckets)
            for i in buckets_averaged:
                i.weight /= max_weight
            while buckets_averaged:
                key = random.randint(0, 10000)
                bucket_number = key % len(buckets_averaged)
                bucket_weight = self.get_weight(buckets_averaged[bucket_number])
                if bucket_weight:
                    hash_value = libscrc.crc32(bytes(key))
                    hash_weight = float(hash_value & 0xffffffff) / 2 ** 32
                    if hash_weight < bucket_weight:
                        # LOG.info(f'nexthop: {buckets_averaged[bucket_number]}')
                        if use_flow_memory:
                            self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)] = buckets_averaged[
                                bucket_number].edge
                        return buckets_averaged[bucket_number].edge
                del buckets_averaged[bucket_number]
        return None
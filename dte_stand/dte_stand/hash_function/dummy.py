from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction
from typing import Generator, Optional, Any, List, Tuple

import logging
LOG = logging.getLogger(__name__)


class DummyHashFunction(BaseHashFunction):
    def _choose_nexthop(self, buckets: List[Bucket]) -> Optional[GraphPathElement]:
        try:
            return buckets[0].edge
        except IndexError:
            # empty list of buckets
            return None
'''
    def __init__(self, *args, **kwargs):
        self._flow_to_bucket_map: dict[Tuple[str, str], GraphPathElement] = {}
        return super().__init__(*args, **kwargs)

    def get_weight(self, bucket):
        return bucket.weight

    def crc16(self, data: bytes, poly=0x8408):
        data = bytearray(data)
        crc = 0xFFFF
        for b in data:
            cur_byte = 0xFF & b
            for _ in range(0, 8):
                if (crc & 0x0001) ^ (cur_byte & 0x0001):
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
                cur_byte >>= 1
        crc = (~crc & 0xFFFF)
        crc = (crc << 8) | ((crc >> 8) & 0xFF)
        return crc & 0xFFFF

    def _choose_nexthop(self, buckets: List[Bucket], flow_id: str) -> Optional[GraphPathElement]:
        if buckets:
            if self._flow_to_bucket_map.get((flow_id, buckets[0].edge.from_)) \
                    and self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)] in list(
                        bucket.edge for bucket in buckets):
                LOG.info(f'nexthop: {self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)]}')
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
                    hash_value = self.crc16(bytes(key))
                    hash_weight = float(hash_value & 0xffffffff) / 2 ** 32
                    if hash_weight < bucket_weight:
                        LOG.info(f'nexthop: {buckets_averaged[bucket_number]}')
                        self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)] = buckets_averaged[
                            bucket_number].edge
                        return buckets_averaged[bucket_number].edge
                del buckets_averaged[bucket_number]
        return None
'''
from dataclasses import dataclass
from typing import List
from ..req import Req


@dataclass
class GroupReqIndexes:
    group_req_id: int
    shm_req_indexes: List[int]
    time_mark: float


@dataclass
class GroupReqObjs:
    group_req_id: int
    shm_req_objs: List[Req]
    time_mark: float

    def to_group_req_index(self):
        return GroupReqIndexes(
            group_req_id=self.group_req_id,
            shm_req_indexes=[req.index_in_shm_mem for req in self.shm_req_objs],
            time_mark=self.time_mark,
        )

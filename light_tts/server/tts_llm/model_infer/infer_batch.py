import time
import torch
import numpy as np
import collections
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from light_tts.common.req_manager import ReqManager
from light_tts.common.mem_manager import MemoryManager
from light_tts.server.core.objs.req import ReqRunStatus
from light_tts.utils.infer_utils import mark_start, mark_end
import numpy as np
import copy
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs import Req, SamplingParams, FinishStatus, ShmReqManager
from light_tts.server.req_id_generator import convert_sub_id_to_group_id


logger = init_logger(__name__)
requests_mapping = {}

@dataclass
class InferenceContext:
    req_manager: ReqManager = None
    shm_req_manager: ShmReqManager = None  # 共享内存请求对象管理
    requests_mapping: Dict[int, "InferReq"] = None
    infer_req_ids = None
    vocab_size = None

    overlap_stream: torch.cuda.Stream = None  # 一些情况下推理进程进行异步折叠操作的异步流对象。

    def register(
        self, req_manager: ReqManager, shm_req_manager: ShmReqManager, vocab_size: int
    ):
        self.req_manager = req_manager
        self.shm_req_manager = shm_req_manager

        self.requests_mapping = {}
        self.group_mapping: Dict[int, InferReqGroup] = {}
        self.infer_req_ids = []

        self.vocab_size = vocab_size
        self.radix_cache = None
        return
    
    def get_overlap_stream(self) -> torch.cuda.Stream:
        if self.overlap_stream is None:
            self.overlap_stream = torch.cuda.Stream()
        return self.overlap_stream

    def add_reqs(self, requests: List[Tuple[int, int]], init_req_obj=True):
        request_ids = []
        for r in requests:

            r_id, r_index = r
            if r_id not in self.requests_mapping.keys():
                r_obj = InferReq(
                    req_id=r_id,
                    req_idx=self.req_manager.alloc(),
                    shm_index=r_index,
                    vocab_size=self.vocab_size,
                )
                self.requests_mapping[r_id] = r_obj
            else:
                r_obj: InferReq = self.requests_mapping[r_id]
                # assert r_obj.paused is True
            

            request_ids.append(r_id)

            if init_req_obj:
                r_obj.init_all()
            r_obj.shm_req.req_status.set_status(ReqRunStatus.RUNNING)

        self.infer_req_ids.extend(request_ids)

        return
    
    def free_a_req_mem(self, free_token_index: List, req: "InferReq", is_group_finished: bool):
        if self.radix_cache is None:
            if is_group_finished:
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][0 : req.cur_kv_len])
            else:
                free_token_index.append(
                    self.req_manager.req_to_token_indexs[req.req_idx][req.shm_req.input_len : req.cur_kv_len]
                )
        else:
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
            if is_group_finished:
                prefix_len = self.radix_cache.insert(key, value)
                old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
                if req.shared_kv_node is not None:
                    assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None
            else:
                free_token_index.append(
                    self.req_manager.req_to_token_indexs[req.req_idx][req.shm_req.input_len : req.cur_kv_len]
                )
                if req.shared_kv_node is not None:
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None
    
    @torch.no_grad()                        
    def filter(self, finished_request_ids: List[int], append_prefill_req_ids: List[int]):
        if len(finished_request_ids) == 0 and len(append_prefill_req_ids) == 0:
            return

        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req: InferReq = self.requests_mapping.pop(request_id)
            group_req_id = convert_sub_id_to_group_id(req.shm_req.request_id)
            if group_req_id in self.group_mapping:
                is_group_finished = self.group_mapping[group_req_id].remove_req(req.shm_req.request_id)
                if is_group_finished:
                    del self.group_mapping[group_req_id]
                self.free_a_req_mem(free_token_index, req, is_group_finished)
            else:
                self.free_a_req_mem(free_token_index, req, True)
            free_req_index.append(req.req_idx)
            # logger.info(f"infer release req_id {req.shm_req.request_id}")
            req.shm_req.shm_infer_released = True
            self.shm_req_manager.put_back_req_obj(req.shm_req)

        if len(finished_request_ids) > 0:
            free_token_index = torch.cat(free_token_index, dim=-1)
            self.req_manager.free(free_req_index, free_token_index)

        finished_req_ids_set = set(finished_request_ids)
        append_prefill_req_ids_set = set(append_prefill_req_ids)

        self.infer_req_ids = [_id for _id in self.infer_req_ids if _id not in finished_req_ids_set and _id not in append_prefill_req_ids_set]

    @torch.no_grad()
    def pause_reqs(self, pause_req_ids: List[int]):
        free_token_index = []
        for request_id in pause_req_ids:
            req: InferReq = self.requests_mapping[request_id]
            self.infer_req_ids.remove(request_id)

            # 不支持多输出的情况的暂停
            self.free_a_req_mem(free_token_index, req, is_group_finished=True)
            req.cur_kv_len = 0
            req.shm_req.shm_cur_kv_len = req.cur_kv_len
            req.paused = True  # 暂停信息标记。

        if len(free_token_index) != 0:
            free_token_index = torch.cat(free_token_index, dim=-1)
            self.req_manager.free_token(free_token_index)

        return self


g_infer_context = InferenceContext()

class InferReqGroup:
    def __init__(
        self,
        group_req_id: int,
    ) -> None:
        self.group_req_id = group_req_id
        self.req_ids_group = []
    
    def get_req(self, index):
        return g_infer_context.requests_mapping[self.req_ids_group[index]]
    
    def get_all_reqs(self):
        return [g_infer_context.requests_mapping[self.req_ids_group[i]] for i in range(len(self.req_ids_group))]

    def add_req(self, req_id):
        self.req_ids_group.append(req_id)

    def remove_req(self, req_id):
        assert req_id in self.req_ids_group
        self.req_ids_group.remove(req_id)
        return len(self.req_ids_group) == 0

    def best_of(self):
        return len(self.req_ids_group)

    def diverse_copy(self, req_manager, is_prefill):
        # record previous status
        prev_req = g_infer_context.requests_mapping[convert_sub_id_to_group_id(self.req_ids_group[0])]
        if prev_req.shared_kv_node is not None:
            prefix_len = prev_req.shared_kv_node.node_prefix_total_len
        else:
            prefix_len = 0
        pre_input_token_ids = prev_req.get_input_token_ids()
        cache_token_id = req_manager.req_to_token_indexs[prev_req.req_idx][prefix_len : len(pre_input_token_ids)]
        # update the InferReq status and mem_manager status for cache sharing
        for req_id in self.req_ids_group[:]:
            if req_id == convert_sub_id_to_group_id(req_id):
                continue
            req = g_infer_context.requests_mapping[req_id]
            req.finish_status.set_status(FinishStatus.NO_FINISH)
            input_token_ids = req.get_input_token_ids()
            req_manager.req_to_token_indexs[req.req_idx][prefix_len : len(input_token_ids)] = cache_token_id
            assert len(input_token_ids) == len(pre_input_token_ids)

class InferSamplingParams:

    def __init__(
        self,
        shm_req: Req,
        vocab_size: int,
    ) -> None:
        self.shm_param = shm_req.sample_params
        if self.shm_param.top_k == -1:
            self.shm_param.top_k = vocab_size
        return

class InferReq:
    def __init__(
        self,
        req_id: int,
        req_idx: int,
        shm_index: int,
        vocab_size: int = -1,
    ):
        self.req_id = req_id
        self.req_idx = req_idx
        self.shm_index = shm_index
        self.vocab_size = vocab_size
        self.initialized = False
        self.paused = False
        self.output_token_ids = []
        self.next_fill_index = -1
    
    def init_all(self):
        if self.initialized is False:
            self.shm_req = g_infer_context.shm_req_manager.get_req_obj_by_index(self.shm_index)
            self.sampling_param: InferSamplingParams = InferSamplingParams(self.shm_req, self.vocab_size)
            self.cur_kv_len = 0
            self.cur_output_len = 0
            self.finish_status = FinishStatus()
            self.semantic_len = self.shm_req.semantic_len
            self.bistream = self.shm_req.bistream
        
        if self.paused or not self.initialized:
            self.shm_req.shm_cur_kv_len = self.cur_kv_len
        
        self.initialized = True
        self.paused = False
        return
    
    def is_uninitialized(self):
        return not self.initialized or self.paused
    
    def get_output_len(self):
        return self.cur_output_len

    def get_cur_total_len(self):
        return self.shm_req.input_len

    def get_input_token_ids(self):
        return list(self.shm_req.prompt_ids[0 : self.get_cur_total_len()])

    def set_next_gen_token_id(self, next_token_id: int, next_token_offset_id: int):
        index = self.get_cur_total_len()
        self.shm_req.prompt_ids[index] = next_token_offset_id
        self.shm_req.input_len += 1
        index = self.get_output_len()
        self.shm_req.output_ids[index] = next_token_id
        return

    def get_last_gen_token(self):
        return self.output_token_ids[-1]

    def update_finish_status(self, eos_ids, fill_token_id=None):
        if (
            self.cur_output_len > 0
            and self.get_last_gen_token() in eos_ids
        ):
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif self.cur_output_len >= self.sampling_param.shm_param.max_new_tokens:
            self.finish_status.set_status(FinishStatus.FINISHED_LENGTH)
        elif self.bistream and self.get_last_gen_token() == fill_token_id:
            self.shm_req.req_status.set_status(ReqRunStatus.WAIT_FOR_TEXT)
        
        return
import asyncio
import multiprocessing.synchronize
import os
import numpy as np
import rpyc
import torch
import traceback
from datetime import timedelta
import torch.multiprocessing as mp
import multiprocessing
from typing import Dict, List, Tuple

from light_tts.server.tts_llm.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs.shm_speech_manager import SharedSpeechManager
from light_tts.server.core.objs import RpcShmParams, RpcShmResults, ShmSyncStatusArray


logger = init_logger(__name__)

class ModelRpcServer:
    def __init__(
        self,
        args,
        rank: int,
        rank_in_node: int,
        node_world_size: int,
        rpc_event: multiprocessing.synchronize.Event,
        rpc_finished_event: multiprocessing.synchronize.Event,
    ):
        super().__init__()
        self.args = args
        self.node_world_size = node_world_size
        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event

        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()
        self.rpc_shm_sync_status = ShmSyncStatusArray(self.node_world_size)
        self.rpc_shm_sync_status.create_or_link_shm()

        self.rank = rank
        self.rank_in_node = rank_in_node
        logger.info(f"Initialized RPC server for rank {self.rank}.")
        return

    def init_model(self, kvargs):
        self.args = kvargs.get("args", None)
        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.load_way = kvargs["load_way"]
        self.style_name = kvargs["style_name"]
        self.mode = kvargs["mode"]
        self.speech_token_size = kvargs["speech_token_size"]
        self.mix_ratio = kvargs.get("mix_ratio", [5, 15])

        self.cache = {}
        self.logger = init_logger(__name__)

        weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]
        
        model_kvargs = {
            "weight_dir": os.path.join(weight_dir, 'CosyVoice-BlankEN'),
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "pt_dir": os.path.join(weight_dir, 'llm.pt'),
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "use_dynamic_prompt_cache": True, # for bistream mode
            "data_type": kvargs.get("data_type", "float16"),
            "style_name": self.style_name,
            "speech_token_size": self.speech_token_size,
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
        }

        self.backend = ContinuesBatchBackend()
        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)        

        return
    
    def prefill(self, reqs):
        try:
            return self.backend.prefill(reqs)
        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Batch prefill encountered an unexpected ERROR: {err_msg}")
            raise e

    def decode(self):
        try:
            return self.backend.decode()
        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Batch decode encountered an unexpected ERROR: {err_msg}")
            raise e

    def pause_reqs(self, req_ids):
        return self.backend.pause_reqs(req_ids)

    def get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()


class ModelRpcClient:
    def __init__(self, model_infer_servers: List[ModelRpcServer], world_size, rpc_event, rpc_finished_event):
        # model_infer_servers 是传入的推理服务对象，但是在重构后，
        # 单卡不使用rpc 通信的时候，里面才有真实对象，当多卡使用rpc
        # 以后，model_infer_servers 传入的是 None 数组
        if world_size == 1:
            self.model_infer_server: ModelRpcServer = model_infer_servers[0]
        else:
            self.model_infer_server: ModelRpcServer = None
        
        self.world_size = world_size
        self.use_rpc = self.world_size != 1
        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()

        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event
        return

    async def init_model(self, kvargs):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("init_model", (kvargs,))
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.init_model(kvargs)
            return

    async def prefill(self, reqs):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("prefill", (reqs,))
            self.rpc_event.set()

            await asyncio.to_thread(self.rpc_finished_event.wait)
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.prefill(reqs)
            return

    async def decode(self):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("decode", ())
            self.rpc_event.set()

            await asyncio.to_thread(self.rpc_finished_event.wait)
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.decode()
            return

    async def pause_reqs(self, req_ids):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("pause_reqs", (req_ids,))
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.pause_reqs(req_ids)
            return
        
    async def get_max_total_token_num(self):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("get_max_total_token_num", ())
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            func_name, ret = self.rpc_shm_results.read_func_result()
            assert func_name == "get_max_total_token_num"
            return ret
        else:
            return self.model_infer_server.get_max_total_token_num()

async def start_model_process(
    args,
    rank,
    rank_in_node,
    node_world_size,
    rpc_event,
    rpc_finished_event,
):
    import light_tts.utils.rpyc_fix_utils as _

    return ModelRpcServer(
        args,
        rank,
        rank_in_node,
        node_world_size,
        rpc_event,
        rpc_finished_event,
    )

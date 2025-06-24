import torch
import numpy as np
from typing import List
from light_tts.server.tts_llm.model_infer.infer_batch import InferReq, g_infer_context
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

#@calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(req_ids: List[int]):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    ignore_eos = []
    bistream_list = []
    win_size = 10
    pad_token = -1
    batch_size = len(req_ids)
    padded_output = torch.full((batch_size, win_size), pad_token, dtype=torch.int64, device='cuda')
    b_ready_cache_len = []
    b_next_fill = []

    for i, request_id in enumerate(req_ids):
        req : InferReq = g_infer_context.requests_mapping[request_id]
        run_reqs.append(req)

        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        
        input_token_ids = req.get_input_token_ids()
        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len
        input_id = input_token_ids[req.cur_kv_len:]
        
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len
        ignore_eos.append((not req.bistream) or (req.shm_req.ignore_eos))
        b_next_fill.append(req.bistream and (req.next_fill_index == len(req.output_token_ids)))
        bistream_list.append(req.bistream)
        output_token_ids = torch.tensor(req.output_token_ids, dtype=torch.int64, device='cuda')
        if len(output_token_ids) > 0:
            length = min(win_size, output_token_ids.shape[0])
            padded_output[i, -length:] = output_token_ids[-length:]
        
    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
    ignore_eos = torch.tensor(ignore_eos, dtype=torch.bool, device='cuda')
    b_next_fill = torch.tensor(b_next_fill, dtype=torch.bool, device='cuda')
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device='cuda')
    bistream_list = torch.tensor(bistream_list, dtype=torch.bool, device='cuda')

    kwargs = {
        "batch_size": batch_size,
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "b_ready_cache_len": b_ready_cache_len,
        "is_prefill": True,
        "output_token_ids": padded_output,
        "ignore_eos": ignore_eos,
        "b_next_fill": b_next_fill,
        "bistream": bistream_list
    }
    
    return kwargs, run_reqs
    
#@calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(req_ids: List[int]):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    ignore_eos = []
    win_size = 10
    pad_token = -1
    batch_size = len(req_ids)
    padded_output = torch.full((batch_size, win_size), pad_token, dtype=torch.int64, device='cuda')
    b_next_fill = []
    bistream_list = []

    for i, request_id in enumerate(req_ids):
        req : InferReq = g_infer_context.requests_mapping[request_id]
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_token_ids = req.get_input_token_ids()
        input_id = input_token_ids[-1]
        seq_len = len(input_token_ids)
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len

        output_token_ids = torch.tensor(req.output_token_ids, dtype=torch.int64, device='cuda')
        length = min(win_size, output_token_ids.shape[0])
        padded_output[i, -length:] = output_token_ids[-length:]

        ignore_eos.append(
            (not req.bistream and (len(req.output_token_ids) < req.sampling_param.shm_param.min_new_tokens))
            or 
            (req.bistream and req.shm_req.ignore_eos)
        )
        b_next_fill.append(req.bistream and (req.next_fill_index == len(req.output_token_ids)) and req.shm_req.ignore_eos)
        bistream_list.append(req.bistream)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
    ignore_eos = torch.tensor(ignore_eos, dtype=torch.bool, device='cuda')
    b_next_fill = torch.tensor(b_next_fill, dtype=torch.bool, device='cuda')
    bistream_list = torch.tensor(bistream_list, dtype=torch.bool, device='cuda')

    kwargs = {
        "batch_size": batch_size,
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "output_token_ids": padded_output,
        "is_prefill": False,
        "ignore_eos": ignore_eos,
        "b_next_fill": b_next_fill,
        "bistream": bistream_list
    }
    return kwargs, run_reqs

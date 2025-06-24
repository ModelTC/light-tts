import triton
import torch
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

from light_tts.utils.tuning_utils import mp_tuning, set_seed, tuning_configs
import sys
import os
import numpy as np

from light_tts.models.llama.triton_kernel.flash_decoding_stage1 import flash_decode_stage1
from light_tts.models.llama.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from light_tts.models.llama.triton_kernel.flash_decoding import LlamaFlashDecodingStage1KernelConfig

@torch.no_grad()
def test_func(
    batch_size,
    max_len,
    head_dim,
    q_head_num,
    kv_head_num,
    dtype,
    test_count: int = 20,
    **run_config,
):
    set_seed()
    tmp_class = type("TestObj", (object,), {})
    infer_state = tmp_class()
    infer_state.batch_size = batch_size
    infer_state.max_len_in_batch = max_len

    infer_state.req_manager = tmp_class()
    infer_state.req_manager.req_to_token_indexs = torch.zeros(
        (infer_state.batch_size, infer_state.max_len_in_batch), dtype=torch.int32, device="cuda"
    )
    infer_state.req_manager.req_to_token_indexs.view(-1)[:] = torch.arange(
        0, infer_state.batch_size * infer_state.max_len_in_batch, step=1, dtype=torch.int32
    ).cuda()
    infer_state.b_req_idx = torch.arange(0, infer_state.batch_size, step=1, dtype=torch.int32).cuda()
    infer_state.b_seq_len = torch.full((infer_state.batch_size,), fill_value=max_len, dtype=torch.int32).cuda()
    infer_state.total_token_num_tensor = torch.sum(infer_state.b_seq_len)

    q = torch.empty((batch_size, q_head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    cache_k = torch.empty((batch_size * max_len, kv_head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    cache_v = torch.empty((batch_size * max_len, kv_head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    mid_o = torch.empty([batch_size, 
                        q_head_num,
                        max_len // run_config["BLOCK_SEQ"] + 1, 
                        head_dim], 
                        dtype=torch.float32, 
                        device="cuda")
    mid_o_logexpsum = torch.empty([batch_size, 
                        q_head_num,
                        max_len // run_config["BLOCK_SEQ"] + 1],
                        dtype=torch.float32, 
                        device="cuda")
    o_tensor = torch.empty_like(q)
    
    fn1 = lambda: flash_decode_stage1(
        q,
        cache_k,
        cache_v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        **run_config
    )

    cost_time1 = triton.testing.do_bench_cudagraph(fn1, rep=test_count)

    fn2 = lambda: flash_decode_stage2(
        mid_o,
        mid_o_logexpsum, 
        infer_state.b_seq_len, 
        o_tensor, 
        **run_config
    )
    cost_time2 = triton.testing.do_bench_cudagraph(fn2, rep=test_count)
    cost_time = cost_time1 + cost_time2

    logger.info(f"bf16 {batch_size, max_len} cost time: {cost_time} ms")
    return cost_time


def get_test_configs(split_id, split_count, **kwargs):
    index = 0
    for BLOCK_SEQ in [16, 32, 64, 128]:
        for BLOCK_N in [8, 16, 32, 64]:
            if BLOCK_SEQ % BLOCK_N != 0:
                continue
            for stage1_num_warps in [1, 2, 4, 8]:
                for stage1_num_stages in [1, 2, 3, 4, 5, 6, 7, 8, 12, 15]:
                    for stage2_num_warps in [1, 2, 4]:
                        for stage2_num_stages in [1, 2, 3, 4]:
                            t_config = {
                                "BLOCK_SEQ": BLOCK_SEQ,
                                "BLOCK_N": BLOCK_N,
                                "stage1_num_warps": stage1_num_warps,
                                "stage1_num_stages": stage1_num_stages,
                                "stage2_num_warps": stage2_num_warps,
                                "stage2_num_stages": stage2_num_stages,
                            }
                        if index % split_count == split_id:
                            yield t_config
                        index += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    import collections

    store_json_ans = collections.defaultdict(dict)

    head_dim = 64
    q_head_num = 14
    kv_head_num = 2

    for seq_len in [64, 128, 256, 512, 1024, 2048]:
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size * seq_len > 32 * 1024:
                continue

            test_func_args = {
                "batch_size": batch_size,
                "max_len": seq_len,
                "head_dim": head_dim,
                "q_head_num": q_head_num,
                "kv_head_num": kv_head_num,
                "dtype": torch.float16,
                "test_count": 20,
            }
            ans = mp_tuning(
                tuning_configs,
                {
                    "test_func": test_func,
                    "test_func_args": test_func_args,
                    "get_test_configs_func": get_test_configs,
                }
            )
            store_json_ans[seq_len][batch_size] = ans
            LlamaFlashDecodingStage1KernelConfig.save_config(
                head_dim=head_dim,
                q_head_num=q_head_num,
                kv_head_num=kv_head_num,
                out_dtype=str(torch.float16),
                store_json_ans=store_json_ans,
            )
import torch
from light_tts.common.kernel_config import KernelConfigs
from functools import lru_cache
from frozendict import frozendict
from typing import Dict

class LlamaFlashDecodingStage1KernelConfig(KernelConfigs):
    kernel_name: str = "triton_flashdecoding"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        batch_size: int,
        avg_seq_len_in_batch: int,
        head_dim: int,
        q_head_num: int,
        kv_head_num: int,
        dtype,
    ) -> dict:
        key_params = {
            "head_dim": head_dim,
            "q_head_num": q_head_num,
            "kv_head_num": kv_head_num,
            "out_dtype": str(dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            batch_size_config: dict = finded_config[
                min(finded_config.keys(), key=lambda x: abs(int(x) - avg_seq_len_in_batch))
            ]
            config = batch_size_config[min(batch_size_config.keys(), key=lambda x: abs(int(x) - batch_size))]
            return config
        else:
            config = {
                "BLOCK_SEQ": 256,
                "BLOCK_N": 16,
                "stage1_num_warps": 1,
                "stage1_num_stages": 2,
                "stage2_num_warps": 4,
                "stage2_num_stages": 2,
            }
            return config
    
    @classmethod
    def save_config(cls, *args, **kwargs) -> None:
        key_params = {
            "head_dim": kwargs["head_dim"],
            "q_head_num": kwargs["q_head_num"],
            "kv_head_num": kwargs["kv_head_num"],
            "out_dtype": kwargs["out_dtype"],
        }
        key_params = frozendict(key_params)

        cls.store_config(key_params, kwargs["store_json_ans"])

def token_decode_attention_flash_decoding(
    q, infer_state, q_head_num, head_dim, cache_k, cache_v, out=None, alloc_tensor_func=torch.empty
):
    batch_size = infer_state.batch_size
    avg_seq_len_in_batch = infer_state.total_token_num // batch_size
    run_config = LlamaFlashDecodingStage1KernelConfig.try_to_get_best_config(
        batch_size, avg_seq_len_in_batch, head_dim, q_head_num, cache_k.shape[1], torch.float16
    )
    BLOCK_SEQ = run_config["BLOCK_SEQ"]

    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .flash_decoding_stage1 import flash_decode_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1], dtype=torch.float32, device="cuda"
    )

    flash_decode_stage1(
        q.view(calcu_shape1),
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
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), **run_config)
    return o_tensor

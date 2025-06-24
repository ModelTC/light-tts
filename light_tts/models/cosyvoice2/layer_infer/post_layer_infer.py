import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from light_tts.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight

from light_tts.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange
from light_tts.models.llama.infer_struct import LlamaInferStateInfo
from light_tts.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from light_tts.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from light_tts.utils.infer_utils import mark_cost_time


class CosyVoice2PostLayerInfer(LlamaPostLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        return

    def _norm(self, input, infer_state, layer_weight: LlamaPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)

    def _slice_get_last_input(self, input_embdings, infer_state: LlamaInferStateInfo):

        if infer_state.is_prefill and infer_state.is_token_healing:
            batch_size = infer_state.batch_size
            b_seq_len_numpy = (infer_state.b_seq_len - infer_state.b_ready_cache_len).detach().cpu().numpy()
            select_index = []
            start_index = 0
            select_token_num = 0
            for cur_len in b_seq_len_numpy:
                select_index.append(start_index + cur_len - 1)
                start_index += cur_len
                select_token_num += 1

            last_index = torch.tensor(select_index, dtype=torch.long, device=input_embdings.device)
            last_input = self.alloc_tensor((select_token_num, self.embed_dim_), dtype=input_embdings.dtype)
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, select_token_num

        if infer_state.is_prefill and not infer_state.return_all_prompt_logics:
            batch_size = infer_state.batch_size
            last_input = self.alloc_tensor((batch_size, self.embed_dim_), dtype=input_embdings.dtype)
            last_index = (
                torch.cumsum(infer_state.b_seq_len - infer_state.b_ready_cache_len, dim=0, dtype=torch.long) - 1
            )
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if infer_state.is_prefill and infer_state.return_all_prompt_logics:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens

        if not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size

        assert False, "Error State"

    def soft_max(self, data):
        return torch.log_softmax(data.permute(1, 0).float(), dim=-1)

    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = last_input.permute(1, 0).view(-1, token_num)
        logic_batch = self.alloc_tensor(
            (layer_weight.llm_decoder_weight_.shape[0], last_input.shape[1]), dtype=last_input.dtype
        )
        torch.addmm(
            layer_weight.llm_decoder_bias_, layer_weight.llm_decoder_weight_, last_input, beta=1.0, alpha=1.0, out=logic_batch
        )

        last_input = None
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor((layer_weight.llm_decoder_weight_.shape[0], token_num), dtype=input_embdings_dtype)
            split_indexes = np.linspace(0, layer_weight.llm_decoder_weight_.shape[0], self.tp_world_size_ + 1, dtype=np.int64)
            dist.all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                logic_batch,
                group=None,
                async_op=False,
            )
        logic_batch = None
        ans_logics = self.alloc_tensor((token_num, layer_weight.llm_decoder_weight_.shape[0]), dtype=torch.float32, is_graph_out=True)
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        torch.log_softmax(ans_logics, dim=-1, out=ans_logics)
        return ans_logics

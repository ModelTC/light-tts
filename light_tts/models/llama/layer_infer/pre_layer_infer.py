import os
import torch
import torch.distributed as dist
import numpy as np

from light_tts.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from light_tts.models.llama.infer_struct import LlamaInferStateInfo
from light_tts.common.basemodel import PreLayerInferTpl
from light_tts.utils.infer_utils import mark_cost_time
from light_tts.models.llama.triton_kernel.embedding import embedding


class LlamaPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        if not self.enable_dp:
            tp_vob_ids = np.linspace(0, network_config["vocab_size"], self.tp_world_size_ + 1, dtype=np.int64)
            self.vob_start_id_, self.vob_end_id_ = int(tp_vob_ids[self.tp_rank_]), int(tp_vob_ids[self.tp_rank_ + 1])
        else:
            self.vob_start_id_, self.vob_end_id_ = 0, network_config["vocab_size"]

        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.tp_world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.tp_world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings

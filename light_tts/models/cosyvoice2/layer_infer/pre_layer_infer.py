import torch
import torch.distributed as dist
import numpy as np

from light_tts.models.cosyvoice2.layer_weights.pre_and_post_layer_weight import CosyVoice2PreAndPostLayerWeight
from light_tts.models.llama.infer_struct import LlamaInferStateInfo
from light_tts.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from light_tts.utils.infer_utils import mark_cost_time
from light_tts.models.llama.triton_kernel.embedding import embedding_tp1

class CosyVoice2PreLayerInfer(LlamaPreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: CosyVoice2PreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.text_llm_audio_emb.shape[1]), dtype=layer_weight.data_type_
        )
        embedding_tp1(input_ids, layer_weight.text_llm_audio_emb, input_embdings)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: CosyVoice2PreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.text_llm_audio_emb.shape[1]), dtype=layer_weight.data_type_
        )
        embedding_tp1(input_ids, layer_weight.text_llm_audio_emb, input_embdings)
        return input_embdings
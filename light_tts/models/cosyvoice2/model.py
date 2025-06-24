import os
import json
import torch

from light_tts.models.cosyvoice2.layer_weights.pre_and_post_layer_weight import CosyVoice2PreAndPostLayerWeight
from light_tts.models.cosyvoice2.layer_infer.pre_layer_infer import CosyVoice2PreLayerInfer
from light_tts.models.cosyvoice2.layer_infer.post_layer_infer import CosyVoice2PostLayerInfer
from light_tts.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from light_tts.models.llama.layer_weights.ds_load_utils import load_ds_weights

from light_tts.models.qwen2.model import Qwen2TpPartModel


class CosyVoice2TpPartModel(Qwen2TpPartModel):
    # weight class
    pre_and_post_weight_class = CosyVoice2PreAndPostLayerWeight

    pre_layer_infer_class = CosyVoice2PreLayerInfer
    post_layer_infer_class = CosyVoice2PostLayerInfer

    def __init__(self, kvargs):
        self.pt_dir = kvargs["pt_dir"]
        self.speech_token_size = kvargs["speech_token_size"]
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        self.text_vob_size = self.config["vocab_size"]


    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(i, torch.float16, network_config=self.config, mode=self.mode, quant_cfg=self.quant_cfg)
            for i in range(self.config["n_layer"])
        ]
        self.weight_dict = torch.load(self.pt_dir, map_location="cpu")
        self.weight_dict = {k.replace("llm.model.", ''): self.weight_dict[k] for k in self.weight_dict.keys()}
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]            
        return 
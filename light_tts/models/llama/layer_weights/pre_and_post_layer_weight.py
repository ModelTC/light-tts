import os
import torch
import numpy as np
from light_tts.common.basemodel import PreAndPostLayerWeight


class LlamaPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.embed_tokens.weight" in weights:
            if self.enable_dp:
                self.wte_weight_ = self._cuda(weights["model.embed_tokens.weight"])
            else:
                self.wte_weight_ = self._cuda(weights["model.embed_tokens.weight"][split_start:split_end, :])
            tie_word_embeddings = self.network_config_.get("tie_word_embeddings", False)
            if tie_word_embeddings:
                self.lm_head_weight_ = self.wte_weight_
        if "lm_head.weight" in weights:
            if self.enable_dp:
                self.lm_head_weight_ = self._cuda(weights["lm_head.weight"])
            else:
                self.lm_head_weight_ = self._cuda(weights["lm_head.weight"][split_start:split_end, :])
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])

        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.wte_weight_, self.lm_head_weight_, self.final_norm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

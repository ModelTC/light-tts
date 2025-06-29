import os
import json
import torch
import math
from light_tts.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from light_tts.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from light_tts.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from light_tts.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from light_tts.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from light_tts.models.llama.layer_weights.ds_load_utils import load_ds_weights
from light_tts.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

from light_tts.models.llama.infer_struct import LlamaInferStateInfo
from light_tts.common.basemodel import TpPartBaseModel
from light_tts.common.mem_utils import select_mem_manager_class
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)


class LlamaTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = LlamaTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        self._reset_num_key_value_heads()
        return

    def _reset_num_key_value_heads(self):
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return

    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"] // self.tp_world_size_,
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"],
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_custom(self):
        """
        模型特殊的一些初始化
        """
        if self.config.get("use_rope_yarn", False):
            self._init_to_get_yarn_rotary()
        elif self.config.get("use_dynamic_ntk", False) or (
            self.config.get("rope_scaling", None) is not None
            and self.config.get("rope_scaling", {}).get("type", "base") == "dynamic"
        ):
            self._init_to_get_dynamic_ntk_rotary()
        elif (
            self.config.get("rope_scaling", None) is not None
            and self.config.get("rope_scaling", {}).get("type", "base") == "su"
        ):
            self._init_to_su_rotary()
        elif (
            self.config.get("rope_scaling", None) is not None
            and self.config.get("rope_scaling", {}).get("rope_type", "base") == "llama3"
        ):
            self._init_to_get_llama3_rotary()
        else:
            self._init_to_get_rotary()
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                mode=self.mode,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["n_layer"])
        ]
        if self.load_way == "HF":
            load_hf_weights(
                self.data_type,
                weight_dir=self.weight_dir_,
                pre_post_layer=self.pre_post_weight,
                transformer_layer_list=self.trans_layers_weight,
                weight_dict=self.weight_dict,
            )
        else:
            load_ds_weights(
                self.data_type,
                weight_dir=self.weight_dir_,
                pre_post_layer=self.pre_post_weight,
                transformer_layer_list=self.trans_layers_weight,
                weight_dict=self.weight_dict,
                prefix="model.layers.",
                num_layer=self.config["n_layer"],
            )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_to_get_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (partial_head_dim / (partial_head_dim - 2)))  # Base change formula
        except:
            pass

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = (
            torch.arange(max(max_seq_len + 1024 * 128, self.max_seq_length), device="cpu", dtype=torch.float32)
            / rope_scaling_factor
        )
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return

    def _init_to_get_dynamic_ntk_rotary(self):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        base = self.config.get("rope_theta", 10000.0)
        if self.config.get("rope_scaling", {}) is None:
            scaling_factor = 1.0
        else:
            scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)
        max_seq_len = max(self.max_seq_length, max_position_embeddings)
        self._cos_cached = torch.zeros((max_seq_len, partial_head_dim // 2), dtype=self.data_type, device="cuda")
        self._sin_cached = torch.zeros((max_seq_len, partial_head_dim // 2), dtype=self.data_type, device="cuda")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached[0:max_position_embeddings, :] = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached[0:max_position_embeddings, :] = torch.sin(freqs).to(self.data_type).cuda()

        for seq_loc_index in range(max_position_embeddings, max_seq_len, 1):
            new_base = base * (
                (scaling_factor * (seq_loc_index + 1) / max_position_embeddings) - (scaling_factor - 1)
            ) ** (partial_head_dim / (partial_head_dim - 2))
            inv_freq = 1.0 / (
                new_base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
            )
            t = torch.tensor(
                [
                    seq_loc_index,
                ],
                device="cpu",
                dtype=torch.float32,
            )
            freqs = torch.outer(t, inv_freq)
            self._cos_cached[seq_loc_index : seq_loc_index + 1, :] = torch.cos(freqs).to(self.data_type).cuda()
            self._sin_cached[seq_loc_index : seq_loc_index + 1, :] = torch.sin(freqs).to(self.data_type).cuda()
        return

    def _init_to_get_yarn_rotary(self):
        from .yarn_rotary_utils import find_correction_range, linear_ramp_mask, get_mscale

        dim = self.head_dim_
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        base = self.config.get("rope_theta", 10000.0)
        if self.config.get("rope_scaling", {}) is None:
            scale = 1.0
        else:
            scale = self.config.get("rope_scaling", {}).get("factor", 1.0)
        original_max_position_embeddings = self.config.get("original_max_position_embeddings", 2048)
        extrapolation_factor = 1.0
        attn_factor = 1.0
        beta_fast = 32.0
        beta_slow = 1.0

        pos_freqs = base ** (torch.arange(0, dim, 2).float().cuda() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, dim // 2).float().cuda()
        ) * extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        mscale = float(get_mscale(scale) * attn_factor)  # Get n-d magnitude scaling corrected for interpolation

        # Build here to make `torch.jit.trace` work.
        max_seq_len_cached = max_position_embeddings
        t = torch.arange(max(max_seq_len_cached, self.max_seq_length), device="cuda", dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(self.data_type).cuda() * mscale
        self._sin_cached = emb.sin().to(self.data_type).cuda() * mscale

        return

    def _init_to_su_rotary(self):
        rope_scaling = self.config["rope_scaling"]
        short_factor = rope_scaling["short_factor"]
        long_factor = rope_scaling["long_factor"]
        original_max_position_embeddings = self.config["original_max_position_embeddings"]
        max_position_embeddings = self.config.get("max_position_embeddings", original_max_position_embeddings)
        base = self.config.get("rope_theta", 10000.0)
        short_factor = torch.tensor(short_factor, dtype=torch.float32, device="cpu")
        long_factor = torch.tensor(long_factor, dtype=torch.float32, device="cpu")

        scale = max_position_embeddings / original_max_position_embeddings
        if scale <= 1.0:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

        max_seq_len = max(self.max_seq_length, max_position_embeddings)
        self._cos_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=self.data_type, device="cuda")
        self._sin_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=self.data_type, device="cuda")

        inv_freq = 1.0 / (
            short_factor
            * base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_)
        )
        t = torch.arange(original_max_position_embeddings, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached[0:original_max_position_embeddings, :] = (
            (torch.cos(freqs) * rope_scaling_factor).to(self.data_type).cuda()
        )
        self._sin_cached[0:original_max_position_embeddings, :] = (
            (torch.sin(freqs) * rope_scaling_factor).to(self.data_type).cuda()
        )

        inv_freq = 1.0 / (
            long_factor
            * base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_)
        )
        t = torch.arange(original_max_position_embeddings, max_seq_len, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached[original_max_position_embeddings:, :] = (
            (torch.cos(freqs) * rope_scaling_factor).to(self.data_type).cuda()
        )
        self._sin_cached[original_max_position_embeddings:, :] = (
            (torch.sin(freqs) * rope_scaling_factor).to(self.data_type).cuda()
        )

        return

    def _init_to_get_llama3_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        base = self.config.get("rope_theta", float(default_base))

        scale_factor = self.config.get("rope_scaling", {}).get("factor", 8.0)
        low_freq_factor = self.config.get("rope_scaling", {}).get("low_freq_factor", 1.0)
        high_freq_factor = self.config.get("rope_scaling", {}).get("high_freq_factor", 4.0)
        origin_context_len = self.config.get("rope_scaling", {}).get("original_max_position_embeddings", 8192)

        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        max_seq_len = max_position_embeddings

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )

        low_freq_wavelen = origin_context_len / low_freq_factor
        high_freq_wavelen = origin_context_len / high_freq_factor
        new_inv_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_inv_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_inv_freqs.append(freq / scale_factor)
            else:
                smooth = (origin_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_inv_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        inv_freq = torch.tensor(new_inv_freqs, dtype=torch.float32, device="cpu")

        t = torch.arange(max(max_seq_len, self.max_seq_length), device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return

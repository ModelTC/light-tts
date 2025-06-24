import os
import torch
import torch.nn.functional as F
from light_tts.common.quantization.quantize_method import QuantizationMethod
from light_tts.common.quantization.registry import QUANTMETHODS
from .fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from .fp8.fp8act_quant_kernel import per_token_group_quant_fp8


class TritonBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from light_tts.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


@QUANTMETHODS.register(["triton-fp8w8a8-block128"])
class TritonFP8w8a8QuantizationMethod(TritonBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.is_moe = False
        self.block_size = 128

    def quantize(self, weight: torch.Tensor):
        # TODO block-wise quant kernel
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None, use_custom_tensor_mananger=True):
        qweight, weight_scale, input_scale = weights
        m, k = input_tensor.shape
        n = qweight.shape[1]
        if input_scale is None:
            input_scale = self.cache_manager.alloc_tensor(
                (m, k // self.block_size), torch.float32, device=input_tensor.device, is_graph_out=False
            )
            input_tensor_q = self.cache_manager.alloc_tensor(
                (m, k), qweight.dtype, device=qweight.device, is_graph_out=False
            )
            per_token_group_quant_fp8(input_tensor, self.block_size, input_tensor_q, input_scale)
        else:
            # TODO
            raise "statci input scale is not supported by triton fp8 block gemm kernel."
        m = input_tensor.shape[0]
        n = qweight.shape[1]
        if out is None:
            if use_custom_tensor_mananger:
                out = self.cache_manager.alloc_tensor(
                    (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
                )
            else:
                out = torch.empty((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        w8a8_block_fp8_matmul(
            input_tensor_q,
            qweight,
            input_scale,
            weight_scale,
            out,
            (self.block_size, self.block_size),
            dtype=input_tensor.dtype,
        )
        return out

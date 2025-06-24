import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM
import argparse
from light_tts.common.build_utils import repair_config
from light_tts.utils.dist_utils import get_current_device_id

data_type_dict = {"float32": 4, "float16": 2, "bfloat16": 2, "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

def get_available_gpu_memory(world_size):
    """
    Get the available GPU memory in MB.
    """
    if world_size == 1:
        device = torch.device("cuda")
        mem = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        return mem
    else:
        mem = 0
        for i in range(world_size):
            device = torch.device(f"cuda:{i}")
            mem += torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        return mem

def get_total_gpu_memory():
    """
    Get the total GPU memory of the machine
    """
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return total_memory / (1024 ** 3)  # Convert to GB
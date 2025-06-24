import torch
import numpy as np
import threading
from light_tts.common.basemodel.layer_weights.meta_weights import BaseWeight
from light_tts.utils.dist_utils import get_current_device_id, get_current_rank_in_dp, get_dp_world_size


class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()
        self.lock = threading.Lock()

    def load_hf_weights(self, weights):
        """
        load weights
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, BaseWeight):
                attr.load_hf_weights(weights)

    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, BaseWeight):
                assert attr.verify_load(), f"Loading {attr_name} of layers {self.layer_num_} fails."

    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(self.data_type_).cuda(get_current_device_id())

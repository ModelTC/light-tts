import torch
from light_tts.common.mem_manager import MemoryManager
from light_tts.common.req_manager import ReqManager


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.b_req_idx = None
        self.b_start_loc = None
        self.b_ready_cache_len = None  # only for prefill prompt cache used.
        self.b_seq_len = None
        # max_len_in_batch prefill 和 decode 阶段含义不同
        # prefill 阶段指每个req 输入token的长度（不包括已经cache的部分）最大值
        # decode 阶段指的是每个req的总长 最大值
        self.max_len_in_batch = None
        self.is_prefill = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_is_contiguous = None
        self.mem_index = None
        self.mem_start = None
        self.mem_end = None
        self.kv_buffer = None

        self.is_token_healing = False
        self.return_all_prompt_logics = False
        self.use_dynamic_prompt_cache = False
        self.multimodal_params = None
        self.is_cuda_graph = False  # 标记是否是cuda graph的捕获推理

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        pass

    def copy_for_cuda_graph(self, new_infer_state):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None:
                    attr_.copy_(attr_value)
        return

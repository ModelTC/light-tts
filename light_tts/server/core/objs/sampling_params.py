import os
import ctypes
from typing import List, Tuple, Union
from light_tts.server.req_id_generator import MAX_BEST_OF

_SAMPLING_EPS = 1e-5
DEFAULT_INPUT_PENALTY = os.getenv("INPUT_PENALTY", "False").upper() in ["ON", "TRUE", "1"]
SKIP_SPECIAL_TOKENS = os.getenv("SKIP_SPECIAL_TOKENS", "True").upper() in ["ON", "TRUE", "1"]

# 从环境变量获取最大长度限制
STOP_SEQUENCE_MAX_LENGTH = int(os.getenv("LIGHTLLM_STOP_SEQUENCE_MAX_LENGTH", 256))
ALLOWED_TOKEN_IDS_MAX_LENGTH = int(os.getenv("LIGHTLLM_ALLOWED_TOKEN_IDS_MAX_LENGTH", 256))
MAX_STOP_SEQUENCES = int(os.getenv("LIGHTLLM_MAX_STOP_SEQUENCES", 10))
REGULAR_CONSTRAINT_MAX_LENGTH = int(os.getenv("LIGHTLLM_REGULAR_CONSTRAINT_MAX_LENGTH", 2048))
GRAMMAR_CONSTRAINT_MAX_LENGTH = int(os.getenv("LIGHTLLM_GRAMMAR_CONSTRAINT_MAX_LENGTH", 2048))
JSON_SCHEMA_MAX_LENGTH = int(os.getenv("LIGHTLLM_JSON_SCHEMA_MAX_LENGTH", 2048))


class AllowedTokenIds(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("ids", ctypes.c_int * ALLOWED_TOKEN_IDS_MAX_LENGTH),
        ("size", ctypes.c_int),
    ]

    def initialize(self, ids: List[int]):
        self.size = len(ids)
        assert self.size <= ALLOWED_TOKEN_IDS_MAX_LENGTH, "Too many allowed token IDs."
        assert all(isinstance(e, int) for e in self.ids), "all must be int"
        self.ids[: self.size] = ids[:]

    def to_list(self):
        return list(self.ids[: self.size])


class ExponentialDecayLengthPenalty(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("item0", ctypes.c_int),
        ("item1", ctypes.c_float),
    ]

    def initialize(self, inputs: Tuple[int, float]):
        assert len(inputs) == 2, "ExponentialDecayLengthPenalty must be Tuple[int, float]"
        self.item0 = inputs[0]
        assert self.item0 >= 0, "ExponentialDecayLengthPenalty item0 must be int >= 0"
        self.item1 = inputs[1]
        assert self.item1 >= 1.0, "ExponentialDecayLengthPenalty item1 must be a float >= 1.0"
        return

    def to_tuple(self):
        return (self.item0, self.item1)


class DecodeNode(ctypes.Structure):
    _fields_ = [
        ("exists", ctypes.c_bool),
        ("node_id_high", ctypes.c_uint64),  # UUID 的高 64 位
        ("node_id_low", ctypes.c_uint64),  # UUID 的低 64 位
        ("ip", ctypes.c_int32 * 4),
        ("rpyc_port", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
    ]

    def initialize(self, data_dict):
        if data_dict is None:
            self.exists = False
            return

        self.exists = True

        pd_node_id = data_dict["node_id"]
        self.node_id_high = (pd_node_id >> 64) & 0xFFFFFFFFFFFFFFFF
        self.node_id_low = pd_node_id & 0xFFFFFFFFFFFFFFFF

        ip_parts = [int(part) for part in data_dict["ip"].split(".")]
        self.ip = (ctypes.c_int32 * 4)(*ip_parts)

        self.rpyc_port = data_dict["rpyc_port"]
        self.max_new_tokens = data_dict["max_new_tokens"]

    def to_dict(self):
        if not self.exists:
            return None
        uuid_int = (self.node_id_high << 64) | self.node_id_low
        return {
            "node_id": uuid_int,
            "ip": ".".join(str(self.ip[i]) for i in range(4)),
            "rpyc_port": self.rpyc_port,
            "max_new_tokens": self.max_new_tokens,
        }


class SamplingParams(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("best_of", ctypes.c_int),
        ("n", ctypes.c_int),
        ("do_sample", ctypes.c_bool),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("ignore_eos", ctypes.c_bool),
        ("max_new_tokens", ctypes.c_int),
        ("min_new_tokens", ctypes.c_int),
        # Whether to count input tokens for presence_penalty, frequency_penalty and repetition_penalty
        ("input_penalty", ctypes.c_bool),
        ("exponential_decay_length_penalty", ExponentialDecayLengthPenalty),
        ("group_request_id", ctypes.c_int),  # p d mode used params
        ("suggested_dp_index", ctypes.c_int),  # suggest dp index, deepseekv2 dp mode, use to suggest used dp_index
        ("move_kv_to_decode_node", DecodeNode),  # move kv to deocde node, only used in pd mode
        ("skip_special_tokens", ctypes.c_bool),  # whether to skip special tokens when decoding
        ("add_special_tokens", ctypes.c_bool),  # whether to add special tokens when encoding
        (
            "add_spaces_between_special_tokens",
            ctypes.c_bool,
        ),  # whether to add spaces between special tokens when decoding
        ("print_eos_token", ctypes.c_bool),  # eos_id will be always ignored except the value is set to True
    ]

    _do_sample: bool = False
    _presence_penalty: float = 0.0
    _frequency_penalty: float = 0.0
    _repetition_penalty: float = 1.0
    _temperature: float = 1.0
    _top_p: float = 1.0
    _top_k: int = -1  # -1 is for all

    def init(self, **kwargs):
        super().__init__()
        self.best_of = kwargs.get("best_of", 1)
        self.n = kwargs.get("n", self.best_of)
        self.do_sample = kwargs.get("do_sample", SamplingParams._do_sample)
        self.presence_penalty = kwargs.get("presence_penalty", SamplingParams._presence_penalty)
        self.frequency_penalty = kwargs.get("frequency_penalty", SamplingParams._frequency_penalty)
        self.repetition_penalty = kwargs.get("repetition_penalty", SamplingParams._repetition_penalty)
        self.temperature = kwargs.get("temperature", SamplingParams._temperature)
        self.top_p = kwargs.get("top_p", SamplingParams._top_p)
        self.top_k = kwargs.get("top_k", SamplingParams._top_k)
        self.ignore_eos = kwargs.get("ignore_eos", False)
        self.max_new_tokens = kwargs.get("max_new_tokens", 16)
        self.min_new_tokens = kwargs.get("min_new_tokens", 1)
        self.input_penalty = kwargs.get("input_penalty", DEFAULT_INPUT_PENALTY)
        self.group_request_id = kwargs.get("group_request_id", -1)
        self.suggested_dp_index = kwargs.get("suggested_dp_index", -1)

        self.skip_special_tokens = kwargs.get("skip_special_tokens", SKIP_SPECIAL_TOKENS)

        self.add_special_tokens = kwargs.get("add_special_tokens", True)
        self.add_spaces_between_special_tokens = kwargs.get("add_spaces_between_special_tokens", True)
        self.print_eos_token = kwargs.get("print_eos_token", False)

        self.exponential_decay_length_penalty = ExponentialDecayLengthPenalty()
        self.exponential_decay_length_penalty.initialize(kwargs.get("exponential_decay_length_penalty", (1, 1.0)))

        self.move_kv_to_decode_node = DecodeNode()
        self.move_kv_to_decode_node.initialize(kwargs.get("move_kv_to_decode_node", None))

        if self.do_sample is False:
            self.temperature = 1.0
            self.top_p = 1.0
            self.top_k = 1
        if (
            self.temperature >= 0.0 and self.temperature < _SAMPLING_EPS
        ):  # temperature is too slow, change to greedy search
            self.temperature = 1.0
            self.top_k = 1

        self.verify()

    @classmethod
    def load_generation_cfg(cls, weight_dir):
        try:
            generation_cfg = {}
            cls._do_sample = generation_cfg.get("do_sample", False)
            cls._presence_penalty = generation_cfg.get("presence_penalty", 0.0)
            cls._frequency_penalty = generation_cfg.get("frequency_penalty", 0.0)
            cls._repetition_penalty = generation_cfg.get("repetition_penalty", 1.0)
            cls._temperature = generation_cfg.get("temperature", 1.0)
            cls._top_p = generation_cfg.get("top_p", 1.0)
            cls._top_k = generation_cfg.get("top_k", -1)
        except:
            pass

    def verify(self):
        if self.best_of <= 0 or self.best_of > MAX_BEST_OF:
            raise ValueError(f"need 0 < best_of <= {MAX_BEST_OF}, but got {self.best_of}")
        if self.n != self.best_of:
            raise ValueError("current only supported n == best_of")
        if self.n <= 0 or self.n > MAX_BEST_OF or self.n > self.best_of:
            raise ValueError(f"need 0 < n <= {MAX_BEST_OF}, n <= {self.best_of}, but got {self.n}")
        if self.presence_penalty < 0.0:
            raise ValueError(f"presence_penalty must >= 0.0, got {self.presence_penalty}")
        if self.frequency_penalty < 0.0:
            raise ValueError(f"frequency_penalty must >= 0.0, got {self.frequency_penalty}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must >= 1.0, got {self.repetition_penalty}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must > 0.0, got {self.temperature}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, got {self.top_k}.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {self.max_new_tokens}.")
        if self.min_new_tokens < 1:
            raise ValueError(f"min_new_tokens must be at least 1, got {self.min_new_tokens}.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                f"min_new_tokens must <= max_new_tokens, but got min {self.min_new_tokens}, max {self.max_new_tokens}."
            )

        return

    def to_dict(self):
        return {
            "do_sample": self.do_sample,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "ignore_eos": self.ignore_eos,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "best_of": self.best_of,
            "input_penalty": self.input_penalty,
        }

    def to_origin_dict(self):
        ret = self.to_dict()
        ret["group_request_id"] = self.group_request_id
        ret["suggested_dp_index"] = self.suggested_dp_index
        return ret

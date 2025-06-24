import json
from functools import lru_cache
from schema import Schema, And, Use, Optional, SchemaError
from light_tts.utils.path_utils import trans_relative_to_abs_path
from light_tts.utils.log_utils import init_logger
from .envs_utils import get_env_start_args
import sys

logger = init_logger(__name__)

def get_config_json(model_dir):
    return {
        "lora_info": [
            {
                "style_name": "CosyVoice2",
            }
        ],
    }

@lru_cache(maxsize=None)
def get_fixed_kv_len():
    start_args = get_env_start_args()
    model_cfg = get_config_json(start_args.model_dir)
    if "prompt_cache_token_ids" in model_cfg:
        return len(model_cfg["prompt_cache_token_ids"])
    else:
        return 0
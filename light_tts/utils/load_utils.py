import os
from hyperpyyaml import load_hyperpyyaml
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../third_party/Matcha-TTS'.format(ROOT_DIR))
# 对于多进程来说不行
def load_yaml(model_dir):
    with open('{}/cosyvoice2.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    return configs
# import faulthandler
# faulthandler.enable()
import os
import numpy as np
import multiprocessing as mp
import threading
from multiprocessing import shared_memory
from light_tts.utils.log_utils import init_logger
from filelock import FileLock
from collections import OrderedDict

logger = init_logger(__name__)


class SharedArray:
    def __init__(self, name, shape, dtype):
        dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        dest_size = np.prod(shape) * dtype_byte_num
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
            logger.info(f"create shm {name}")
        except Exception as e:
            shm = shared_memory.SharedMemory(name=name, create=False, size=dest_size)
            logger.info(f"link shm {name} error {str(e)}")
        
        if shm.size != dest_size:
            logger.info(f"size not same, unlink shm {name} and create again")
            shm.unlink()
            shm.close()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
                logger.info(f"create shm {name}")
            except Exception as e:
                shm = shared_memory.SharedMemory(name=name, create=False, size=dest_size)
                logger.info(f"link shm {name} error {str(e)}")

        self.shm = shm  # SharedMemory 对象一定要被持有，否则会被释放
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)


class SharedTensorManager:
    def __init__(self, name, size) -> None:
        self.name = name
        self.size = size
        self.shape_infs = SharedArray(f"{name}_shapes", (size, 2), dtype=np.int32)
        self.tensors = [None for _ in range(size)]
        return
    
    def set_index_data(self, index, shape, data, dtype):
        shm_arr = SharedArray(f"{self.name}_{index}_tensor", shape, dtype=dtype)
        shm_arr.arr[:, :] = data
        self.shape_infs.arr[index,:] = shape
        self.tensors[index] = shm_arr
        return
    
    def get_index_tensor_shape(self, index):
        return tuple(self.shape_infs.arr[index])
    
    def get_index_tensor(self, index, dtype):
        shape = self.get_index_tensor_shape(index)
        shm_arr = SharedArray(f"{self.name}_{index}_tensor", shape, dtype=dtype)
        self.tensors[index] = shm_arr
        return shm_arr
    
    # def release(self, index):
    #     shape = self.get_index_tensor_shape(index)
    #     shm_arr = SharedArray(f"{self.name}_{index}_tensor", shape, dtype=np.float16)
    #     shm_arr.shm.unlink() # 销毁shm。
    #     shm_arr.shm.close()
    #     logger.info(f"release shm tensor index {index}")
    #     return


class SharedSpeechManager:
    def __init__(self, name, size, init_mark=True) -> None:
        self.name = name
        self.size = size
        self.use_marks = SharedArray(f"{name}_use_marks", (size,), dtype=np.int32)
        if init_mark:
            self.use_marks.arr[:] = 0
        self.lru_cache = OrderedDict()
        self.lock = threading.Lock()

        self.prompt_speech_16k_manager = SharedTensorManager(f"{name}_prompt_speech_16k", size)
        self.speech_feat_manager = SharedTensorManager(f"{name}_speech_feat", size)
        self.speech_token_manager = SharedTensorManager(f"{name}_speech_token", size)
        self.spk_embedding_manager = SharedTensorManager(f"{name}_spk_embedding", size)
        return
        
    
    def alloc(self, speech_md5):
        with self.lock:
            if speech_md5 in self.lru_cache:
                self.lru_cache.move_to_end(speech_md5)
                return self.lru_cache[speech_md5], True
            index = None
            if len(self.lru_cache) >= self.size:
                key, value = self.lru_cache.popitem(last=False)
                index = value
            else:
                for i in range(self.size):
                    if self.use_marks.arr[i] == 0:
                        index = i
                        break
            
            if index is None:
                raise RuntimeError("alloc error")

            self.use_marks.arr[index] = 1
            self.lru_cache[speech_md5] = index
            return index, False
    
    def set_index_data(self, index, shape, data):
        self.prompt_speech_16k_manager.set_index_data(index, shape, data, np.float32)
        self.use_marks.arr[index] = 2
        return

    def get_index_data(self, index):
        if self.use_marks.arr[index] >= 2:
            return self.prompt_speech_16k_manager.get_index_tensor(index, dtype=np.float32)
        return None

    def set_index_speech(self, index, speech_token, speech_feat, spk_embedding):
        self.speech_token_manager.set_index_data(index, speech_token.shape, speech_token, np.int32)
        self.speech_feat_manager.set_index_data(index, speech_feat.shape, speech_feat, np.float32)
        self.spk_embedding_manager.set_index_data(index, spk_embedding.shape, spk_embedding, np.float32)
        self.use_marks.arr[index] = 3
        return

    def get_index_speech_token(self, index):
        if self.use_marks.arr[index] >= 3:
            return self.speech_token_manager.get_index_tensor(index, dtype=np.int32)
        return None

    def get_index_speech(self, index):
        if self.use_marks.arr[index] >= 3:
            return self.speech_token_manager.get_index_tensor(index, dtype=np.int32), self.speech_feat_manager.get_index_tensor(index, dtype=np.float32), self.spk_embedding_manager.get_index_tensor(index, np.float32)
        return None

    def speech_data_ready(self, index):
        if self.use_marks.arr[index] >= 3:
            return True
        return False
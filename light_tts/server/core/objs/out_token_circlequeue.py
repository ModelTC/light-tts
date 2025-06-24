import os
import ctypes
from typing import Tuple

import numpy as np

LIGHTTTS_STREAM_AUDIO_MAX_SIZE = int(os.getenv("LIGHTTTS_STREAM_AUDIO_MAX_SIZE", 65536))
LIGHTLLM_OUT_TOKEN_QUEUE_SIZE = int(os.getenv("LIGHTLLM_OUT_TOKEN_QUEUE_SIZE", 6))


class QueueItem(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("data", ctypes.c_float * LIGHTTTS_STREAM_AUDIO_MAX_SIZE),
        ("data_len", ctypes.c_int),
        ("token_offset", ctypes.c_int),
        ("finalize", ctypes.c_bool),
    ]

    def __init__(self):
        self.data_len = 0
        self.token_offset = 0
        self.finalize = False

    def set(self, tts_speech: np.ndarray, token_offset: int, finalize: bool):
        assert len(tts_speech) <= LIGHTTTS_STREAM_AUDIO_MAX_SIZE

        nbytes = tts_speech.nbytes
        ctypes.memmove(self.data, tts_speech.ctypes.data, nbytes)
        self.data_len = len(tts_speech)
        self.token_offset = token_offset
        self.finalize = finalize
        return

    def get(self):
        return (
            np.ctypeslib.as_array(self.data)[:self.data_len],
            self.token_offset,
            self.finalize
        )


class CircularQueue(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("items", QueueItem * LIGHTLLM_OUT_TOKEN_QUEUE_SIZE),  # 循环队列的元素
        ("head", ctypes.c_int),  # 指向队列头部
        ("tail", ctypes.c_int),  # 指向队列尾部
    ]

    def __init__(self):
        # 初始化头和尾
        self.head = 0
        self.tail = 0

    def is_empty(self):
        return self.head == self.tail

    def is_full(self):
        return (self.tail + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE == self.head

    def push(self, tts_speech: np.ndarray, token_offset: int, finalize: bool = False):
        if self.is_full():
            raise Exception("Queue is full")

        # 添加元素
        item: QueueItem = self.items[self.tail]
        item.set(tts_speech, token_offset, finalize)

        # 更新尾部
        self.tail = (self.tail + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE

    def pop(self) -> Tuple[np.ndarray, int, bool]:
        if self.is_empty():
            raise Exception("Queue is empty")

        # 移除元素
        item: QueueItem = self.items[self.head]
        result = item.get()

        # 更新头部
        self.head = (self.head + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE
        return result

    def peek(self) -> Tuple[np.ndarray, int, bool]:
        if self.is_empty():
            raise Exception("Queue is empty")

        item: QueueItem = self.items[self.head]
        result = item.get()
        return result

    def pop_no_ret(self) -> None:
        """
        移除一个对象，但是不返回， 与peek配合使用
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        self.head = (self.head + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE
        return

    def __len__(self):
        # 计算当前元素数量
        return (self.tail - self.head + LIGHTLLM_OUT_TOKEN_QUEUE_SIZE) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE

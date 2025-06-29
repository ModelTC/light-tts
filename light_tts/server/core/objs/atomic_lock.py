import atomics
from multiprocessing import shared_memory
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)


class AtomicShmLock:
    def __init__(self, lock_name: str):
        self.lock_name = lock_name
        self.dest_size = 4
        self._init_shm()

        if self.shm.size != self.dest_size:
            logger.info(f"size not same, unlink lock shm {lock_name} and create again")
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self._init_shm()
        self.shm.buf.cast("i")[0] = 0
        return

    def _init_shm(self):
        try:
            shm = shared_memory.SharedMemory(name=self.lock_name, create=True, size=self.dest_size)
            logger.info(f"create lock shm {self.lock_name}")
        except:
            shm = shared_memory.SharedMemory(name=self.lock_name, create=False, size=self.dest_size)
            logger.info(f"link lock shm {self.lock_name}")
        self.shm = shm
        return

    def __enter__(self):
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(0, 1):
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(1, 0):
                pass
        return False

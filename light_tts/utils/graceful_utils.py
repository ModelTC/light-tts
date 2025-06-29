import sys
from light_tts.utils.log_utils import init_logger
logger = init_logger(__name__)


def graceful_registry(sub_module_name):
    import signal
    # 子进程在受到 SIGTERM的时候，不能自己就提前退出。
    def graceful_shutdown(signum, frame):
        logger.info(f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown...")
        if signum == signal.SIGTERM:
            logger.info(f"{sub_module_name} recive sigterm")

    signal.signal(signal.SIGTERM, graceful_shutdown)
    return

import socket
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

def alloc_can_use_network_port(num=3, used_nccl_port=None):
    port_list = []
    for port in range(20000, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(("localhost", port))
            if result != 0 and port != used_nccl_port:
                port_list.append(port)

            if len(port_list) == num:
                return port_list
    return None

class PortLocker:
    def __init__(self, ports):
        self.ports = ports
        self.sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(len(self.ports))]
        for _socket in self.sockets:
            _socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def lock_port(self):
        for _socket, _port in zip(self.sockets, self.ports):
            try:
                _socket.bind(("", _port))
                _socket.listen(1)
            except Exception as e:
                logger.error(f"port {_port} has been used")
                raise e

    def release_port(self):
        for _socket in self.sockets:
            _socket.close()

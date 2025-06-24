from .continues_batch.impl import ContinuesBatchQueue


def build_req_queue(args, router, dp_size_in_node: int = 1):
    queue_class = None
    queue_class = ContinuesBatchQueue

    if dp_size_in_node == 1:
        return queue_class(args, router, 0, dp_size_in_node)
from typing import List, Dict
from light_tts.server.core.objs import Req

class DecodeReq:
    def __init__(
        self,
        req: Req,
        decode_token_hop_len: int,
        flow_pre_lookahead_len: int,
        eos_id: int = 0,
    ) -> None:
        self.request_id = req.request_id
        self.req = req
        self.decode_token_hop_len = decode_token_hop_len
        self.flow_pre_lookahead_len = flow_pre_lookahead_len
        self.eos_id = eos_id
        self.finish_stream = False
        self.token_offset = 0
    
    def get_infer_data(self):
        output_len = self.req.get_output_len()
        if not self.req.stream:
            output_ids = list(self.req.output_ids[:output_len])
            finalize = self.req.finish_status.is_finished()
        else:
            this_token_hop_len = self.decode_token_hop_len + self.req.prompt_token_pad if self.token_offset == 0 else self.decode_token_hop_len
            offset = this_token_hop_len + self.flow_pre_lookahead_len + self.token_offset
            if output_len >= offset:
                output_ids = list(self.req.output_ids[:offset])
                finalize = False
            elif self.req.finish_status.is_finished():
                output_ids = list(self.req.output_ids[:output_len])
                finalize = True
        
        if output_ids[-1] == self.eos_id:
            output_ids.pop()
        speech_index = self.req.speech_index
        request_id = self.request_id
        token_offset = self.token_offset
        return output_ids, speech_index, request_id, token_offset, finalize
    
    def update_one_decode(self, finalize: bool):
        if self.req.stream:
            this_token_hop_len = self.decode_token_hop_len + self.req.prompt_token_pad if self.token_offset == 0 else self.decode_token_hop_len
            self.token_offset += this_token_hop_len
        self.finish_stream = finalize

    def out_queue_is_full(self):
        return self.req.out_tokens_queue.is_full()
    
    def can_set_release_mark(self):
        if self.req.is_aborted:
            return True
        if self.req.finish_status.is_finished() and self.finish_stream:
            return True
        return False
import torch
from typing import List
from light_tts.common.basemodel.triton_kernel.apply_penalty import apply_penalty
from dataclasses import dataclass
from ...infer_batch import InferReq, g_infer_context


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25, not_first=False):
    sorted_probs, sorted_idx = weighted_scores.softmax(dim=-1).sort(descending=True, stable=True, dim=-1)
    top_k = min(top_k, sorted_probs.shape[1])
    sorted_probs = sorted_probs[:, :top_k]
    sorted_idx = sorted_idx[:, :top_k]

    probs_sum = sorted_probs.cumsum(dim=-1)
    mask = probs_sum >= top_p
    mask[:, 0] = False
    if not_first:
        mask[:, 1] = False
    sorted_probs[mask] = 0.0

    sampled_index = sorted_probs.multinomial(1, replacement=True)
    batch_next_token_ids = torch.gather(sorted_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids

def random_sampling(weighted_scores):
    top_ids = weighted_scores.softmax(dim=-1).multinomial(1, replacement=True).view(-1)
    return top_ids

def ras_sampling(weighted_scores, decoded_tokens, top_p=0.8, top_k=25, win_size=10, tau_r=0.1, not_first=False):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k, not_first=not_first)
    rep_num = (decoded_tokens[:,-win_size:].to(weighted_scores.device) == top_ids.unsqueeze(1)).sum(dim=-1)

    mask = rep_num >= win_size * tau_r
    if mask.any():
        top_ids[mask] = random_sampling(weighted_scores[mask])
    return top_ids

def sample(
    weighted_scores: torch.Tensor,
    decoded_tokens: torch.Tensor,
    ignore_eos: torch.Tensor,
    bistream: torch.Tensor,
    eos_id: int = 6561,
    fill_token_id: int = 6563,
):
    num_trials, max_trials = 0, 100
    
    top_ids = ras_sampling(weighted_scores, decoded_tokens)
    mask = (top_ids > eos_id) | ((top_ids == eos_id) & ignore_eos)
    while mask.any():
        top_ids[mask] = ras_sampling(weighted_scores[mask], decoded_tokens[mask], not_first=True)
        # top_ids 为大于speech_token_size的值，而且不是fill_token，或者是fill_token，但不是(bistream且在text没receiving完的情况)
        mask = ((top_ids > eos_id) & (top_ids != fill_token_id | ~bistream | ~ignore_eos)) | ((top_ids == eos_id) & ignore_eos)

        num_trials += 1
        # TODO: 错误处理
        if num_trials > max_trials:
            raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input {}!'.format(max_trials, mask))

    return top_ids

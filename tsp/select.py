"""
Functions to decide the next node selection
given a distribution over a problem's nodes.
"""

import torch

from tsp.utils import batch_select_gather


def sample_select(problems, log_probs):
    probs = torch.exp(log_probs)
    next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    next_sel = batch_select_gather(problems, next_idx)
    return next_idx, next_sel


def greedy_select(problems, log_probs):
    next_idx = torch.argmax(log_probs, dim=1)
    next_sel = batch_select_gather(problems, next_idx)
    return next_idx, next_sel

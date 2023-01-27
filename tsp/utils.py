import torch
from torch.distributions.categorical import Categorical
import numpy as np
import pkg_resources
import random

from tsp.datagen import TspDataset


def get_coords(batch_size, problem_size):
    """Get single-tensor problems dataset with shape (N, S, 2)."""
    dataset = TspDataset(size=problem_size, num_samples=batch_size)
    return torch.stack(dataset[:], dim=0)


def get_costs(problems, select_idxs=None):
    """
    Adapted from code provided with the paper "Attention, Learn to Solve Routing Problems!"
    https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/problem_tsp.py

    Modified to allow not passing selection indices, in which case we assume the problems stack
    contains solutions where nodes are already sorted in order of intended travel.
    """
    if select_idxs is not None:
        # Check that tours are valid, i.e. contain 0 to n -1
        all_valid_tours = (
            torch.arange(select_idxs.size(1), out=select_idxs.data.new())
            .view(1, -1)
            .expand_as(select_idxs)
            == select_idxs.data.sort(1)[0]
        ).all()

        if not all_valid_tours:
            raise ValueError("Invalid tour(s) given by 'select_idxs'")

        # Gather solution in order of tour
        s = problems.gather(1, select_idxs.unsqueeze(-1).expand_as(problems))
    else:
        s = problems

    # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    return (s[:, 1:] - s[:, :-1]).norm(p=2, dim=2).sum(1) + (s[:, 0] - s[:, -1]).norm(
        p=2, dim=1
    )


def all_to(tensors, device="cpu"):
    """Move all tensors to device at once."""
    on_device = [tensor.to(device) for tensor in tensors]
    return on_device


def batch_select_gather(problems, selection_idxs):
    """
    Extracts coordinates from problems at selection_idxs.
    Expects problems to have shape (N, S, 2)
    and selection_idxs to have shape (N,).
    """
    sel_idx_g = torch.stack([selection_idxs] * problems.shape[-1], dim=-1).unsqueeze(1)
    return torch.gather(problems, dim=1, index=sel_idx_g)  # (N, 1, 2)


def batch_dist_gather(distributions, batch_select_idxs):
    """
    Extracts batched distribution probabilities (or
    log probs) at the indices specified by
    batch_select_idxs. Expects distributions to have
    shape (N, S, S) and batch_select_idxs to have
    shape (N, S).
    """
    select_idxs_g = batch_select_idxs.unsqueeze(dim=-1)
    return torch.gather(distributions, dim=-1, index=select_idxs_g).squeeze(-1)


def get_grad_norm(parameters):
    """Get the gradient (vector) L2 norm from parameters iterable."""
    grad_params = filter(lambda p: p.grad is not None, parameters)
    flat_grads = [p.grad.flatten() for p in grad_params]
    grads_vec = torch.cat(flat_grads, dim=0)
    norm = torch.norm(grads_vec, p=2)
    return norm.item()


def float_equality(x, y, eps=1e-4):
    return (x < y + eps) and (x > y - eps)


def get_entropy(log_probs):
    """
    Returns entropy for each decision.
    Assumes log_probs has shape (N, S, S).
    Returns entropy matrix of shape (N, S),
    computing over the last 'S' dim.
    """
    assert log_probs.shape[1] == log_probs.shape[2]
    dist = Categorical(probs=torch.exp(log_probs))
    return dist.entropy()


def is_installed(module_name):
    installs = [pkg.key for pkg in pkg_resources.working_set]
    return module_name in installs


def bool_to_additive_mask(mask):
    mask = (
        mask.float()
        .masked_fill(mask == 0, float(0.0))
        .masked_fill(mask == 1, float("-inf"))
    )
    return mask


def generate_square_subsequent_mask(sz):
    """
    Taken directly from PyTorch docs:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def seed_rand_gen(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

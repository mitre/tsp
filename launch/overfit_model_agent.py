"""
Optimization debug script for testing model convergence
using supervised examples.

E.g. TspMontyStyleModel does not overfit on a single tour 
example when training using REINFORCE. This script attempts 
to see if it can overfit in the supervised case.
"""

import torch
import torch.nn.functional as F
import numpy as np

from tsp.utils import get_coords
from tsp.model.monty_style import TspMontyStyleModel
from tsp.agent import TspAgent


# overfit data parameters
problem_size = 10
num_tours = 1

# initialize model and agent
model = TspMontyStyleModel(dim_model=128, num_enc_layers=3, num_dec_layers=3)
agent = TspAgent(model)

# initialize optimizer (no RL algorithm)
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

# get tour data and arbitrary labels
problems = get_coords(num_tours, problem_size)

labels = [
    torch.tensor(np.random.choice(problem_size, size=problem_size, replace=False))
    for _ in range(num_tours)
]
labels = torch.stack(labels, dim=0)
flat_labels = labels.flatten()

# attempt to overit to labels with NLL loss
while True:
    _, sel_idxs, log_probs = agent(problems)
    incorr_pred = (sel_idxs != labels).sum(dim=-1)
    flat_log_probs = log_probs.view(-1, problem_size)

    # replace -inf with a approximation in prob space (nll_loss becomes inf otherwise)
    inf_approx_mask = torch.full_like(flat_log_probs, -100)
    adj_log_probs = torch.where(
        flat_log_probs == float("-inf"), inf_approx_mask, flat_log_probs
    )

    loss = F.nll_loss(adj_log_probs, flat_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"{loss.item():5.2f} {incorr_pred.float().mean().item():5.2f}")

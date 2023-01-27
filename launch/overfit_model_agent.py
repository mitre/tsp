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

from tsp.datagen import OracleDataset
from tsp.utils import get_coords
from tsp.model.monty_style import TspMontyStyleModel
from tsp.agent import TspAgent


# overfit data parameters
problem_size = 5
num_tours = 1

# initialize model and agent
model = TspMontyStyleModel(dim_model=128, num_enc_layers=3, num_dec_layers=3)
agent = TspAgent(model)

# initialize optimizer (no RL algorithm)
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

dataset = OracleDataset(
    "example_shuffle_data.npy",
    "example_shuffle_labels.npy",
)

# get tour data and arbitrary labels
problems, labels = dataset[0]
problems = problems.unsqueeze(0)
labels = labels.unsqueeze(0)
flat_labels = labels.flatten()

# attempt to overit to labels with NLL loss
while True:
    _, _, log_probs = agent.use(problems, labels)
    flat_log_probs = log_probs.view(-1, problem_size)

    loss = F.nll_loss(flat_log_probs, flat_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"{loss.item():5.2f}")

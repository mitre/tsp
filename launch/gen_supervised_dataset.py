"""
Generates and writes to disk a datset of optimal tours
using TspOracleAgent. Coordinates are saved in optimal
selection order (with arbitrary first choice) so that
no other data needs to be saved.

Note this may be too slow unless an external solver is
used (e.g. PyConcorde).
"""

import time
import os
import os.path as osp
import torch
import numpy as np

import tsp
from tsp.agent import TspOracleAgent
from tsp.utils import get_coords, seed_rand_gen


# run parameters
problem_size = 50
num_tours = int(1e6)
batch_size = 100
save_id = ""  # leave empty for no extra description suffix in dataset name

# dataset gen code
agent = TspOracleAgent()

seed_rand_gen(1234)

print(
    f"Generating and saving solutions for {num_tours} tours of size {problem_size}..."
)
before = time.time()

problems = get_coords(num_tours, problem_size)
batches = torch.split(problems, batch_size, dim=0)

solutions = []
for batch in batches:
    b_solutions, _ = agent.solve(batch)
    solutions.append(b_solutions)

solutions = torch.cat(solutions, dim=0)

print(f"...done, {time.time() - before:.2f}s")

# write array of solutions to disk
save_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "datasets")
if not osp.exists(save_dir):
    os.makedirs(save_dir)

file_name = f"solutions_{problem_size}n_{num_tours}t"
if save_id:
    file_name += "_" + save_id
file_name += ".npy"

save_path = osp.join(save_dir, file_name)

with open(save_path, "wb") as f:
    np.save(f, solutions.numpy())

print(f"Saved dataset @ {save_path}")

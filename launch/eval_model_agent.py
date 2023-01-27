"""
For evaluating a model-based agent's
performance on TSP problems of a fixed length.
"""

import torch
import os.path as osp

import tsp
from tsp.model.monty_style import TspMontyStyleModel, TspMsAcModel
from tsp.agent import TspAgent
from tsp.eval import evaluate_agent
from tsp.utils import seed_rand_gen
from tsp.select import sample_select, greedy_select


# TSP parameters
problem_size = 5
num_tours_eval = int(1e6)
batch_size = num_tours_eval // (2 * problem_size)

# model and agent setup
force_cuda_idx = None  # leave None for first avaialble idx (or cpu)
model_base = "size5_patch_a2c_ms_long/params_62500.pt"
model = TspMsAcModel(128, 6, 6, 6)
select_fn = sample_select
best_of = 1

agent = TspAgent(model)
agent.set_select(select_fn)

if force_cuda_idx is not None:
    agent.to(force_cuda_idx)

model_path = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "runs", model_base)
state_dict = torch.load(model_path, map_location=agent.device)
agent.load_state_dict(state_dict)

agent.eval_mode()

seed_rand_gen(1234)

# run evals
print(
    f"Evaluating agent on {num_tours_eval} tours of size {problem_size}, taking best of {best_of}..."
)
min_cost, max_cost, avg_cost, std_cost = evaluate_agent(
    agent, problem_size, num_tours_eval, batch_size, best_of=best_of
)
print("...done")
print("Tour cost summary:")
print(f"\tMin - {min_cost:5.2f}")
print(f"\tMax - {max_cost:5.2f}")
print(f"\tAvg - {avg_cost:5.2f}")
print(f"\tStd - {std_cost:5.2f}")

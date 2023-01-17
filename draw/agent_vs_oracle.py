"""
Plot example tours from agent predictions (left subplots).
These are compared to the exact solutions (right subplots).
"""

import os
import os.path as osp
from matplotlib import pyplot as plt
import torch

import tsp
from tsp.model.monty_style import TspMontyStyleModel, TspMsAcModel
from tsp.agent import TspAgent, TspOracleAgent
from tsp.utils import get_coords

from base import plot_tsp


# TSP parameters and save id
problem_size = 20
num_tours_eval = 10
save_id = ""  # leave empty for no extra description suffix in png name

# model and agent setup
model_base = "size20_a2c_ms_eval/params_50000.pt"
model = TspMsAcModel(128, 6, 6, 6)

model_agent = TspAgent(model)
model_path = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "runs", model_base)
state_dict = torch.load(model_path, map_location=model_agent.device)
model_agent.load_state_dict(state_dict)

oracle_agent = TspOracleAgent()

# find predicted and exact solutions
print(f"Computing {num_tours_eval} solutions each, problem size {problem_size}")
problems = get_coords(num_tours_eval, problem_size)
print(f"Running model...")
_, pred_idxs = model_agent.solve(problems)
print(f"Running exact solver...")
_, exact_idxs = oracle_agent.solve(problems)

# plot agent vs oracle (left and right, respectively)
print("Plotting...")
fig, axes = plt.subplots(
    num_tours_eval, 2, sharey=True, figsize=(10, 5 * num_tours_eval)
)
for tour_idx, (model_ax, oracle_ax) in enumerate(axes):
    plot_tsp(problems[tour_idx], pred_idxs[tour_idx], model_ax)
    plot_tsp(problems[tour_idx], exact_idxs[tour_idx], oracle_ax)
fig.suptitle("Agent (Left) vs Oracle (Right)", fontsize=16)

# save as PNG and show plot
# matplotlib may change aspect ratios for large plots,
# where the PNG should have aspect ratios preserved
save_dir = osp.join(osp.dirname(__file__), "saved")
if not osp.exists(save_dir):
    os.makedirs(save_dir)

file_name = f"agent_vs_oracle_{problem_size}n_{num_tours_eval}t"
if save_id:
    file_name += "_" + save_id
file_name += ".png"

save_path = osp.join(save_dir, file_name)
fig.savefig(save_path)
print(f"Saved fig @ {save_path}")

plt.show()

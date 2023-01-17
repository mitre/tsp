"""
For evaluating random performance on
TSP problems of a fixed length.
"""

from tsp.agent import TspRandomAgent
from tsp.eval import evaluate_agent
from tsp.utils import seed_rand_gen


# TSP parameters
problem_size = 5
num_tours_eval = int(1e6)

# eval random agent
agent = TspRandomAgent()

seed_rand_gen(1234)

print(f"Evaluating random agent on {num_tours_eval} tours of size {problem_size}...")
min_cost, max_cost, avg_cost, std_cost = evaluate_agent(
    agent, problem_size, num_tours_eval  # no batching, using CPU memory
)
print("...done")
print("Tour cost summary:")
print(f"\tMin - {min_cost:5.2f}")
print(f"\tMax - {max_cost:5.2f}")
print(f"\tAvg - {avg_cost:5.2f}")
print(f"\tStd - {std_cost:5.2f}")

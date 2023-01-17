"""
For evaluating oracle (exact solution) performance 
on TSP problems of a fixed length.

Supports worker subprocesses to speed this up.
But can specify '0' workers for serial eval in main thread,
which is useful for comparison to other serial evals when 
using the same random seed.

WARNING: parallel eval is currently unreliable when using
    Concorde, and so this feature is disabled for right now.
"""

import time

from tsp.agent import TspOracleAgent
from tsp.eval import parallel_eval_agent, evaluate_agent
from tsp.utils import seed_rand_gen


# TSP parameters
problem_size = 50
num_tours_eval = int(1e6)

# eval parameters
batch_size = num_tours_eval
num_workers = 0  # use 0 for serial eval (PARAM IGNORED WHEN USING CONCORDE)

# eval oracle agent
agent = TspOracleAgent()

seed_rand_gen(1234)

print(f"Evaluating oracle agent on {num_tours_eval} tours of size {problem_size}...")
before = time.time()

if num_workers > 0 and not agent.use_concorde:
    min_cost, max_cost, avg_cost, std_cost = parallel_eval_agent(
        agent, problem_size, num_tours_eval, batch_size, num_workers
    )
else:
    assert num_workers == 0
    min_cost, max_cost, avg_cost, std_cost = evaluate_agent(
        agent, problem_size, num_tours_eval, batch_size
    )

print(f"...done, {time.time() - before:.2f}s")
print("Tour cost summary:")
print(f"\tMin - {min_cost:5.2f}")
print(f"\tMax - {max_cost:5.2f}")
print(f"\tAvg - {avg_cost:5.2f}")
print(f"\tStd - {std_cost:5.2f}")

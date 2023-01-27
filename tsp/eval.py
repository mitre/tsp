import torch
from itertools import permutations
import multiprocessing as mp
import copy
import math
from tqdm import tqdm

from tsp.utils import get_coords, get_costs, batch_select_gather, seed_rand_gen
from tsp.datagen import TspDataset


def batched_eval(agent, problems, batch_size=None):
    """
    Generate solutions and return corresponding costs.
    Allows for batched forward pass over problems, which is useful for large
    evaluation routines which need to happen on GPU.
    Omit 'batch_size' param to generate all tour costs in one batch.
    """

    # Why do we even have the dataset class if we are going to use it like
    # this??
    if isinstance(problems, TspDataset):
        problems = torch.stack(problems.data, 0)

    if batch_size is None:
        batch_size = len(problems)

    batches = torch.split(problems, batch_size, dim=0)

    costs = torch.empty(0)
    for batch in batches:
        b_solutions, _ = agent.solve(batch)
        b_costs = get_costs(b_solutions)
        costs = torch.cat((costs, b_costs), dim=0)

    return costs


def batched_eval_repeat(agent, problems, repeats, batch_size=None):
    """
    Like 'batched_eval' but repeated 'repeats' times for each problem.
    Output costs have shape (repeats, num_samples, problem_size, 2).
    """
    assert type(repeats) is int and repeats >= 1
    num_problems = problems.shape[0]

    problems = torch.cat([problems] * repeats, dim=0)
    costs = batched_eval(agent, problems, batch_size)
    split_costs = torch.split(costs, num_problems, dim=0)
    repeat_costs = torch.stack(split_costs, dim=0)

    return repeat_costs


def evaluate_agent(agent, problem_size, num_samples, batch_size=None, best_of=1):
    """
    Evaluate agent on generated problems with fixed number of nodes.
    Omit 'batch_size' param to run all 'num_samples' evalutations in one batch.

    Setting 'best_of' > 1 repeats each problem that many times and reports
    the lowest costs across these repetitions. This is only useful
    if the agent selection strategy is non-deterministic.
    """
    problems = get_coords(num_samples, problem_size)

    if best_of > 1:
        repeat_costs = batched_eval_repeat(
            agent, problems, best_of, batch_size=batch_size
        )
        costs, _ = torch.min(repeat_costs, dim=0)
    else:
        costs = batched_eval(agent, problems, batch_size)

    min_cost = costs.min().item()
    max_cost = costs.max().item()
    avg_cost = costs.mean().item()
    std_cost = costs.std().item()

    return min_cost, max_cost, avg_cost, std_cost


def parallel_eval_agent(agent, problem_size, num_samples, batch_size, num_workers):
    """
    TODO
    """
    assert num_workers >= 1

    def worker_routine(barrier, queue, seed, itr, agent, problem_size, batch_size):
        seed_rand_gen(seed)
        for _ in range(itr):
            problems = get_coords(batch_size, problem_size)
            solutions, _ = agent.solve(problems)
            costs = get_costs(solutions)
            queue.put(costs)
        barrier.wait()  # not waiting for main process here breaks queue, but may be a better way to do this

    queue = mp.Queue()
    barriers = [mp.Barrier(2) for _ in range(num_workers)]
    itr = math.ceil(num_samples / (batch_size * num_workers))

    worker_kwargs = [
        dict(
            barrier=barriers[idx],
            queue=queue,
            seed=idx + 1,
            itr=itr,
            agent=copy.deepcopy(agent),  # expensive if model is large
            problem_size=problem_size,
            batch_size=batch_size,
        )
        for idx in range(num_workers)
    ]

    workers = [
        mp.Process(target=worker_routine, kwargs=wk_kwargs)
        for wk_kwargs in worker_kwargs
    ]

    for p in workers:
        p.start()

    costs = torch.empty(0)
    for _ in tqdm(range(num_workers * itr)):
        sample = queue.get()
        costs = torch.cat((costs, sample), dim=0)

    for idx, p in enumerate(workers):
        barriers[idx].wait()
        p.join()

    min_cost = costs.min().item()
    max_cost = costs.max().item()
    avg_cost = costs.mean().item()
    std_cost = costs.std().item()

    return min_cost, max_cost, avg_cost, std_cost


def oracle(problem):
    """
    Returns exact solution with corresponding
    tour indices and cost. Note this is O(n!)
    with problem size!

    Only accepts a single problem since
    this is SLOW and there is no numpy or
    torch support for batched permutations.
    Use sparingly.
    """
    assert len(problem.shape) == 2  # S, 2
    assert isinstance(problem, torch.Tensor)

    problem_size = len(problem)
    if problem_size > 10:
        print(
            "WARNING: attempting exact solve of TSP using brute-force method "
            f"with large problem size = {problem_size}"
        )
        print("This becomes exceptionally slow after around 10 nodes")

    perms = torch.tensor(list(permutations(torch.arange(problem_size).tolist())))

    problem_stack = torch.stack([problem] * len(perms), dim=0)
    problem_perm_list = [
        batch_select_gather(problem_stack, perms[:, step_idx])
        for step_idx in range(problem_size)
    ]
    problem_perms = torch.cat(problem_perm_list, dim=1)

    costs = get_costs(problem_perms)
    min_cost, min_idx = torch.min(costs, dim=0)

    best_tour = problem_perms[min_idx]
    best_perm = perms[min_idx]

    return best_tour, best_perm, min_cost

import torch
import numpy as np

from tsp.select import sample_select
from tsp.utils import all_to, batch_select_gather, is_installed
from tsp.eval import oracle
from tsp.bridge import ConcordeSolver


class TspAgent:
    """
    TODO
    """

    def __init__(self, model, use_available_device=True):
        use_gpu = torch.cuda.is_available() and use_available_device
        self.device = torch.device("cuda" if use_gpu else "cpu")

        self.model = model
        self.model.to(self.device)

        self.select_fn = sample_select  # defaults to sampling next node

    def __call__(self, problems):
        """
        Get predicted solutions and log-probs for optimization.
        Expects problems to have shape (N, S, 2), where
        'N' is the batch dim and 'S' is the problem
        sequence dim.
        """
        problems = problems.to(self.device)
        model_outputs = self.model(problems, self.select_fn)
        return all_to(model_outputs, device="cpu")

    @torch.no_grad()
    def solve(self, problems):
        """
        Get predicted solutions and solution indices only.
        WARNING: Assumes these are the first two return vals from model __call__!
        """
        problems = problems.to(self.device)
        selections, select_idxs = self.model(problems, self.select_fn)[:2]
        return all_to((selections, select_idxs), device="cpu")

    def to(self, cuda_idx):
        """Manually set cuda idx (e.g. when using several GPUs)."""
        self.device = torch.device(cuda_idx)
        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def set_select(self, select_fn):
        self.select_fn = select_fn

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


class TspRandomAgent:
    """
    Randomly re-sorts problems.
    No model required.
    """

    def __call__(self, problems):
        batch_size, problem_size = problems.shape[:-1]

        count_list = []
        for _ in range(batch_size):
            count = np.arange(problem_size)
            np.random.shuffle(count)
            count_list.append(count)
        select_idxs = torch.tensor(np.stack(count_list, axis=0))

        sel_list = [
            batch_select_gather(problems, select_idxs[:, step_idx])
            for step_idx in range(problem_size)
        ]
        selections = torch.cat(sel_list, dim=1)

        return selections, select_idxs

    def solve(self, problems):
        return self.__call__(problems)


class TspOracleAgent:
    """
    Provides exact solutions to TSP problems.

    Uses Concorde (via pyconcorde) if installed.
    Otherwise defaults to a brute-force O(n!)
    search, which is intractable for problem
    sizes not much greater than 10.
    """

    def __init__(self, force_using_brute=False):
        if force_using_brute or not is_installed("pyconcorde"):
            # use slow (maybe intractable) brute force solver
            self.use_concorde = False
        else:
            self.concorde_solver = ConcordeSolver()
            self.use_concorde = True

    def __call__(self, problems):
        if self.use_concorde:
            return self._run_concorde(problems)
        else:
            return self._batched_brute_force(problems)

    def solve(self, problems):
        solutions, solution_idxs, _ = self.__call__(problems)
        return solutions, solution_idxs

    def _run_concorde(self, problems):
        return self.concorde_solver(problems)

    def _batched_brute_force(self, problems):
        """
        Wraps oracle eval to simulate batched
        solution generation.
        """
        batch_size, _ = problems.shape[:-1]

        if batch_size > 1:
            prob_list = torch.split(problems, 1)
            prob_list = [split.squeeze(0) for split in prob_list]
        else:
            prob_list = [problems.squeeze(0)]

        results = [oracle(problem) for problem in prob_list]
        sol_tup, sol_idx_tup, cost_tup = zip(*results)

        solutions = torch.stack(sol_tup, dim=0)
        solution_idxs = torch.stack(sol_idx_tup, dim=0)
        costs = torch.stack(cost_tup, dim=0)

        return solutions, solution_idxs, costs

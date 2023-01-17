import unittest
import torch

from tsp.model.monty_style import TspMontyStyleModel
from tsp.agent import TspAgent, TspRandomAgent
from tsp.select import sample_select, greedy_select
from tsp.utils import get_coords, batch_select_gather


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.problems = get_coords(batch_size=32, problem_size=10)
        self.model = TspMontyStyleModel()

    def _basic_solution_assertions(self, solutions):
        self.assertEqual(solutions.shape, self.problems.shape)

        # quick way to check that the coordinate sets are equal
        # but not necessarily in the same order
        # this isn't fool-proof but almost certainly checks what we want
        self.assertFalse(torch.allclose(solutions, self.problems))
        self.assertTrue(torch.allclose(solutions.sum(dim=1), self.problems.sum(dim=1)))

    def _basic_select_idxs_assertions(self, select_idxs, solutions):
        batch_size, problem_size = self.problems.shape[:-1]
        self.assertEqual(select_idxs.shape, self.problems.shape[:-1])

        min_compare = torch.zeros(batch_size)
        max_compare = torch.full((batch_size,), problem_size - 1, dtype=torch.float)
        count_compare = torch.stack(
            [torch.arange(problem_size)] * batch_size, dim=0
        )  # 0, 1, 2 ... for each batch slice
        self.assertTrue(
            torch.allclose(select_idxs.min(dim=1)[0].to(torch.float), min_compare)
        )
        self.assertTrue(
            torch.allclose(select_idxs.max(dim=1)[0].to(torch.float), max_compare)
        )
        self.assertTrue(
            torch.allclose(
                select_idxs.sort(dim=1)[0].to(torch.float),
                count_compare.to(torch.float),
            )
        )

        sel_ordered_problem = [
            batch_select_gather(self.problems, select_idxs[:, problem_idx])
            for problem_idx in range(problem_size)
        ]

        sel_ordered_problem = torch.cat(sel_ordered_problem, dim=1)

        self.assertTrue(torch.allclose(sel_ordered_problem, solutions))

    def _basic_prob_assertions(self, log_probs):
        batch_size, problem_size = self.problems.shape[:-1]
        self.assertEqual(
            log_probs.shape, torch.empty(batch_size, problem_size, problem_size).shape
        )
        self.assertTrue(
            torch.allclose(
                torch.exp(log_probs).sum(dim=-1), torch.ones(batch_size, problem_size)
            )
        )

    def test_agent_basics(self):
        agent = TspAgent(self.model)

        agent.set_select(sample_select)
        solutions, select_idxs, log_probs = agent(self.problems)
        self._basic_solution_assertions(solutions)
        self._basic_select_idxs_assertions(select_idxs, solutions)
        self._basic_prob_assertions(log_probs)

        agent.set_select(greedy_select)
        solutions, select_idxs, log_probs = agent(self.problems)
        self._basic_solution_assertions(solutions)
        self._basic_select_idxs_assertions(select_idxs, solutions)
        self._basic_prob_assertions(log_probs)

    def test_solve(self):
        agent = TspAgent(self.model)
        solutions, _ = agent.solve(self.problems)
        self._basic_solution_assertions(solutions)

    def test_rand_agent(self):
        rand_agent = TspRandomAgent()
        solutions, select_idxs = rand_agent(self.problems)
        self._basic_solution_assertions(solutions)
        self._basic_select_idxs_assertions(select_idxs, solutions)

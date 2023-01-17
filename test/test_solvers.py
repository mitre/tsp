"""
For testing external solvers, if installed.
E.g. pyconcorde
"""

import unittest
from functools import wraps
import torch
import numpy as np
import os
import math

from tsp.utils import is_installed, get_coords, float_equality
from tsp.eval import oracle
from tsp.bridge import ConcordeSolver, c_stdout_redirector
from tsp.agent import TspOracleAgent


# decorator which blocks running a test when the specified submodule is not installed
def run_when_installed(module_name):
    def wrap_construct(func):
        @wraps(func)
        def wrapper(testcase_obj):
            if is_installed(module_name):
                func(testcase_obj)

        return wrapper

    return wrap_construct


class TestPyconcorde(unittest.TestCase):
    def _check_concorde_sorts(self, xs, ys, expected_cost=None):
        """
        Assert Pyconcorde generates a tour in sorted order, starting
        at index 0 each time (e.g. 0, 1, 2, 3 or 0, 3, 2, 1 for n=4).
        This is Concorde's solution pattern when the optimal tour
        is the same order as the input nodes.

        Can also provide an expected cost to verify.
        """
        from concorde.tsp import TSPSolver

        assert len(xs) == len(ys)

        with open(os.devnull, "w+b") as nullout:
            with c_stdout_redirector(nullout):
                solver = TSPSolver.from_data(xs, ys, "EUC_2D")
                solution = solver.solve()

        # concorde always starts at index 0
        expected_solA = np.arange(len(xs))
        expected_solB = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.flip(expected_solA)[:-1]), axis=0
        )

        self.assertTrue(
            np.allclose(solution.tour, expected_solA)
            or np.allclose(solution.tour, expected_solB)
        )

        if expected_cost is not None:
            self.assertTrue(float_equality(solution.optimal_value, expected_cost))

    def _compare_concorde_solver_w_oracle(self, problem):
        concorde = ConcordeSolver()
        _, _, conc_cost = concorde(problem)

        _, _, brute_cost = oracle(problem.squeeze())

        self.assertTrue(torch.allclose(conc_cost, brute_cost))

    @run_when_installed("pyconcorde")
    def test_pyconcorde_directly(self):
        """
        Tests pyconcorde without using tsp.bridge.ConcordeSolver wrapper.

        Illustrates Concorde's weird rounding behavior on some tests,
        where the cost of each edge is rounded to the nearest whole number.

        This results in reporting incorrect costs when any edge is not an
        integer, and finding suboptimal tours (!!!), especially when any
        coordinate is not composed of integer (but can occur even if all
        are integers...).
        """
        # edge length = round(1.0) -> 1.0 x 8 edges = 8 cost
        xs = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0])
        ys = np.asarray([0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0])
        self._check_concorde_sorts(xs, ys, 8.0)

        # edge length = round(sqrt(2.0)) -> 1.0 x 8 edges = 8 cost
        xs = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
        ys = np.asarray([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0])
        self._check_concorde_sorts(xs, ys, 8.0)

        # edge length = round(1.7) -> 2.0 x 8 edges = 16 cost
        xs = np.asarray([0.0, 0.0, 0.0, 1.7, 3.4, 3.4, 3.4, 1.7])
        ys = np.asarray([0.0, 1.7, 3.4, 3.4, 3.4, 1.7, 0.0, 0.0])
        self._check_concorde_sorts(xs, ys, 16.0)

    @run_when_installed("pyconcorde")
    def test_concorde_solver_int_data(self):
        for _ in range(100):
            problem = get_coords(1, 5)
            problem = torch.round(problem * 100)

            self._compare_concorde_solver_w_oracle(problem)

    @run_when_installed("pyconcorde")
    def test_concorde_solver_float_data(self):
        for _ in range(100):
            problem = get_coords(1, 5)
            self._compare_concorde_solver_w_oracle(problem)

    @run_when_installed("pyconcorde")
    def test_concorde_oracle_agent(self):
        problems = get_coords(100, 5)

        conc_agent = TspOracleAgent()
        _, _, conc_costs = conc_agent(problems)

        brute_agent = TspOracleAgent(force_using_brute=True)
        _, _, brute_costs = brute_agent(problems)

        self.assertTrue(torch.allclose(conc_costs, brute_costs))


if __name__ == "__main__":  # TODO remove
    unittest.main()

import unittest
import torch
from torch.nn.utils import clip_grad_norm_

from tsp.model.monty_style import TspMontyStyleModel, TspMsAcModel
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C
from tsp.utils import get_coords, get_grad_norm, float_equality


def run_basic_loss_checks(tcase, problems, agent, algo):
    agent_outputs = agent(problems)
    loss = algo.loss(*agent_outputs)
    tcase.assertEqual(len(loss.shape), 0)

    for p in tcase.agent.parameters():
        tcase.assertIsNone(p.grad)

    tcase.optimizer.zero_grad()
    loss.backward()

    for p in tcase.agent.parameters():
        tcase.assertIsNotNone(p.grad)


class TestReinforce(unittest.TestCase):
    def setUp(self):
        self.problems = get_coords(batch_size=32, problem_size=10)
        self.pad_problems = get_coords(batch_size=32, problem_size=range(5, 11))
        self.model = TspMontyStyleModel()
        self.agent = TspAgent(self.model)
        self.optimizer = torch.optim.Adam(self.agent.parameters())

    def test_loss_fixed_seq(self):
        algo = TspReinforce(self.optimizer)
        run_basic_loss_checks(self, self.problems, self.agent, algo)

    def test_loss_pad_seq(self):
        algo = TspReinforce(self.optimizer)
        run_basic_loss_checks(self, self.pad_problems, self.agent, algo)

    def test_grad_clip(self):
        algo_no_clip = TspReinforce(self.optimizer)
        algo_with_clip = TspReinforce(self.optimizer, grad_norm_clip=1.0)

        def compute_grad_norm(problems, agent, algo):
            agent_outputs = agent(problems)
            loss = 1e3 * algo.loss(*agent_outputs)
            algo.optimizer.zero_grad()
            loss.backward()
            grad_norm_before_clip = clip_grad_norm_(
                agent.parameters(), algo.grad_norm_clip
            ).item()
            grad_norm_after_clip = get_grad_norm(agent.parameters())
            algo.optimizer.zero_grad()
            return grad_norm_before_clip, grad_norm_after_clip

        grad_norm_before_no_clip, grad_norm_after_no_clip = compute_grad_norm(
            self.problems, self.agent, algo_no_clip
        )
        self.assertTrue(
            float_equality(grad_norm_before_no_clip / grad_norm_after_no_clip, 1.0)
        )

        grad_norm_before_clip, grad_norm_after_clip = compute_grad_norm(
            self.problems, self.agent, algo_with_clip
        )
        self.assertGreater(grad_norm_before_clip, grad_norm_after_clip)
        self.assertTrue(float_equality(grad_norm_after_clip, 1.0))

    def test_optimize(self):
        algo = TspReinforce(self.optimizer)
        algo.optimize_agent(self.agent, self.problems)
        # no assertions for now, just verifying nothing breaks


class TestA2C(unittest.TestCase):
    def setUp(self):
        self.problems = get_coords(batch_size=32, problem_size=10)
        self.pad_problems = get_coords(batch_size=32, problem_size=range(5, 11))
        self.model = TspMsAcModel()
        self.agent = TspAgent(self.model)
        self.optimizer = torch.optim.Adam(self.agent.parameters())

    def test_loss_fixed_seq(self):
        algo = TspA2C(self.optimizer)
        run_basic_loss_checks(self, self.problems, self.agent, algo)

    def test_loss_pad_seq(self):
        algo = TspA2C(self.optimizer)
        run_basic_loss_checks(self, self.pad_problems, self.agent, algo)


if __name__ == "__main__":
    from tsp.logger import Logger

    Logger.dummy_init()
    torch.autograd.set_detect_anomaly(True)
    unittest.main()

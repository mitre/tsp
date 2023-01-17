import unittest
import torch
from torch.nn.utils import clip_grad_norm_

from tsp.model.monty_style import TspMontyStyleModel, TspMsAcModel
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C
from tsp.utils import get_coords, get_grad_norm, float_equality


class TestReinforce(unittest.TestCase):
    def setUp(self):
        self.problems = get_coords(batch_size=32, problem_size=10)
        self.model = TspMontyStyleModel()
        self.agent = TspAgent(self.model)
        self.optimizer = torch.optim.Adam(self.agent.parameters())

    def test_loss(self):
        algo = TspReinforce(self.optimizer)

        agent_outputs = self.agent(self.problems)
        loss = algo.loss(*agent_outputs)
        self.assertEqual(len(loss.shape), 0)

        for p in self.agent.parameters():
            self.assertIsNone(p.grad)

        self.optimizer.zero_grad()
        loss.backward()

        for p in self.agent.parameters():
            self.assertIsNotNone(p.grad)

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
        self.model = TspMsAcModel()
        self.agent = TspAgent(self.model)
        self.optimizer = torch.optim.Adam(self.agent.parameters())

    def test_loss(self):
        algo = TspA2C(self.optimizer)

        agent_outputs = self.agent(self.problems)
        loss = algo.loss(*agent_outputs)
        self.assertEqual(len(loss.shape), 0)

        for p in self.agent.parameters():
            self.assertIsNone(p.grad)

        self.optimizer.zero_grad()
        loss.backward()

        for p in self.agent.parameters():
            self.assertIsNotNone(p.grad)

import unittest
import math
import torch

from tsp.utils import get_costs, float_equality


class TestUtils(unittest.TestCase):
    def test_get_costs(self):
        p0 = torch.tensor([0.0, 0.0])
        p1 = torch.tensor([0.0, 1.0])
        p2 = torch.tensor([1.0, 1.0])
        p3 = torch.tensor([1.0, 0.0])

        # square tour
        prob_0 = torch.stack([p1, p2, p0, p3], dim=0).unsqueeze(0)
        pi_0 = torch.tensor([2, 0, 1, 3]).unsqueeze(0)
        self.assertEqual(get_costs(prob_0, pi_0).item(), 4.0)

        # zig-zag tour
        prob_1 = torch.stack([p0, p1, p2, p3], dim=0).unsqueeze(0)
        pi_1 = torch.tensor([1, 3, 0, 2]).unsqueeze(0)
        cost_1 = get_costs(prob_1, pi_1).item()
        float_compare = 2.0 + 2 * math.sqrt(2)
        self.assertTrue(float_equality(cost_1, float_compare))

        # batch of both tours
        prob_stack = torch.cat([prob_0, prob_1], dim=0)
        pi_stack = torch.cat([pi_0, pi_1], dim=0)
        cost_stack = get_costs(prob_stack, pi_stack)
        compare_stack = torch.tensor([4.0, float_compare])
        self.assertTrue(torch.allclose(cost_stack, compare_stack))

        # only passing problems (computes tour length in order given)
        cost_stack = get_costs(prob_stack)  # no pi provided
        compare_stack = torch.tensor([float_compare, 4.0])
        self.assertTrue(torch.allclose(cost_stack, compare_stack))

        # invalid tour indices
        pi_invalid = torch.tensor([1, 3, 0, 1]).unsqueeze(0)
        pi_stack = torch.cat([pi_0, pi_invalid], dim=0)
        self.assertRaises(ValueError, get_costs, prob_stack, pi_stack)

import unittest
import torch
import numpy as np
from math import sqrt

from tsp.utils import get_costs, float_equality, perm_shuffle
from tsp.datagen import SENTINEL


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
        float_compare = 2.0 + 2 * sqrt(2)
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

        # batch of tours with different sequence length (sentinel-padded)
        sntnl = torch.tensor([SENTINEL, SENTINEL])

        prob_a = torch.stack([p1, p2, p0, p3, p1, sntnl], dim=0).unsqueeze(0)
        prob_b = torch.stack([p1, p2, p0, p3, sntnl, sntnl], dim=0).unsqueeze(0)
        prob_c = torch.stack([p1, p2, p0, p3, p0, p2], dim=0).unsqueeze(0)
        prob_stack = torch.cat([prob_a, prob_b, prob_c], dim=0)

        pi_a = torch.tensor([2, 0, 1, 3, 4, int(SENTINEL)]).unsqueeze(0)
        pi_b = torch.tensor([1, 3, 0, 2, int(SENTINEL), int(SENTINEL)]).unsqueeze(0)
        pi_c = torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0)
        pi_stack = torch.cat([pi_a, pi_b, pi_c], dim=0)

        cost_stack = get_costs(prob_stack, pi_stack)
        compare_stack = torch.tensor(
            [4.0 + sqrt(2), 2.0 + 2 * sqrt(2), 4.0 + 2 * sqrt(2)]
        )
        self.assertTrue(torch.allclose(cost_stack, compare_stack))

        # repeat but only passing problems
        cost_stack = get_costs(prob_stack)
        compare_stack = torch.tensor(
            [2.0 + 2 * sqrt(2), 2.0 + 2 * sqrt(2), 4.0 + 2 * sqrt(2)]
        )
        self.assertTrue(torch.allclose(cost_stack, compare_stack))

    def test_perm_relabel(self):
        b = 32
        n = 10

        problems = np.random.rand(b, n, 2)

        def get_perm_stack(b, n):
            ps = [np.random.permutation(n) for _ in range(b)]
            return np.stack(ps, axis=0)

        labels = get_perm_stack(b, n)
        perms = get_perm_stack(b, n)

        shuf_probs, relabels = perm_shuffle(problems, labels, perms)

        for idx in range(b):
            prob, lbl, perm = problems[idx], labels[idx], perms[idx]
            sprob, slbl = shuf_probs[idx], relabels[idx]

            self.assertTrue(np.allclose(prob[lbl], prob[perm][slbl]))
            self.assertTrue(np.allclose(sprob, prob[perm]))


if __name__ == "__main__":
    unittest.main()

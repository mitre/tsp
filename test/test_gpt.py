import unittest
import torch

from tsp.model.gpt_style import *


class TestGPT(unittest.TestCase):
    def test_before_start_token_mask(self):
        
        start_token = torch.zeros(2)

        x = torch.rand(10, 10, 2)
        
        token_loc = torch.tensor([1, 9, 4, 5, 2, 3, 0, 8, 3, 9])
        for i in range(10):
            x[i, token_loc[i], :] = start_token

        mask = before_start_token_mask(x, start_token, include_token=False)
        self.assertTrue((torch.sum(mask, dim=-1) == token_loc).all())

        for i in range(10):
            self.assertFalse(mask[i, token_loc[i]])
            for j in range(token_loc[i]):
                self.assertTrue(mask[i, j])

        mask = before_start_token_mask(x, start_token, include_token=True)
        self.assertTrue((torch.sum(mask, dim=-1) - 1 == token_loc).all())

        for i in range(10):
            self.assertTrue(mask[i, token_loc[i]])
            for j in range(token_loc[i]):
                self.assertTrue(mask[i, j])

    def test_attention_mask(self):
        
        x = torch.rand(10, 10, 2)
        mask = attention_mask(x)
        for i in range(10):
            self.assertTrue(mask[i, i])
            for j in range(i):
                self.assertTrue(mask[i, j])

    def test_model(self):
        data = torch.FloatTensor(2, 5, 2).uniform_(0, 1)
        dec = GPTDecoder(8, 1)

        data = dec.append_start_token(data)

        dec(data)


if __name__ == '__main__':
    unittest.main()

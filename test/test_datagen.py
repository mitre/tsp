import unittest
import torch
import numpy as np

from tsp.datagen import TspDataset, SENTINEL


class TestDatagen(unittest.TestCase):

    # SLOW
    # def test_default(self):
    #     dataset = TspDataset()
    #     self.assertEqual(len(dataset), int(1e6))
    #     self.assertEqual(dataset[0].shape, torch.empty(50, 2).shape)
    #     self.assertTrue(torch.all(dataset[0] >= 0))
    #     self.assertTrue(torch.all(dataset[0] <= 1))

    def test_fixed_size_datagen(self):
        dataset = TspDataset(size=10, num_samples=100)
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset[0].shape, torch.empty(10, 2).shape)
        self.assertTrue(torch.all(dataset[0] >= 0))
        self.assertTrue(torch.all(dataset[0] <= 1))

        data = torch.stack(dataset.data, dim=0)
        self.assertFalse(torch.any(data == SENTINEL))

    def test_varying_size_datagen(self):
        dataset = TspDataset(size=range(5, 11), num_samples=100)
        self.assertEqual(len(dataset), 100)

        data = torch.stack(dataset.data, dim=0)
        num_sentinels = (data[:, :, 0] == SENTINEL).sum(dim=1)

        self.assertLessEqual(num_sentinels.max(), 5)
        self.assertTrue(
            torch.allclose((data[:, :, 1] == SENTINEL).sum(dim=1), num_sentinels)
        )

        for idx in range(len(data)):
            compare = data[idx].clone()

            if num_sentinels[idx] > 0:
                compare[-num_sentinels[idx] :] = SENTINEL

            self.assertTrue(torch.allclose(data[idx], compare))

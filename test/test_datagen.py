import unittest
import torch

from tsp.datagen import TspDataset


class TestDatagen(unittest.TestCase):

    # SLOW
    # def test_default(self):
    #     dataset = TspDataset()
    #     self.assertEqual(len(dataset), int(1e6))
    #     self.assertEqual(dataset[0].shape, torch.empty(50, 2).shape)
    #     self.assertTrue(torch.all(dataset[0] >= 0))
    #     self.assertTrue(torch.all(dataset[0] <= 1))

    def test_small_custom(self):
        dataset = TspDataset(size=10, num_samples=100)
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset[0].shape, torch.empty(10, 2).shape)
        self.assertTrue(torch.all(dataset[0] >= 0))
        self.assertTrue(torch.all(dataset[0] <= 1))

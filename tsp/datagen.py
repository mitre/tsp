"""
Creates 2D Traveling Salesman Problems as a stack of coordinates uniformly
sampled in range [0, 1] for both dimensions.

Adapted from code provided with the paper "Attention, Learn to Solve Routing Problems!"
https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/problem_tsp.py
"""

from torch.utils.data import Dataset
import torch
import os
import pickle


class TspDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=int(1e6), offset=0):
        super(TspDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    torch.FloatTensor(row)
                    for row in (data[offset : offset + num_samples])
                ]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TspRedundantDataset(Dataset):
    """
    Creates TspDataset where one problem is randomly
    generated, then all other problems are identical
    repeats of that tour.

    Useful for isolating model performance with respect
    to number of sampled action rollouts per optimization
    step when increasing batch size. With TspDataset,
    increasing the batch size also averages the loss
    over distinct problems.
    """

    def __init__(self, size=10, num_samples=512):
        self.data = [torch.FloatTensor(size, 2).uniform_(0, 1)] * num_samples
        self.size = len(self.data)

        print("Getting exact solution for single problem in TspRedundantDataset...")
        from tsp.eval import oracle

        self.best_tour, self.best_perm, self.min_cost = oracle(self.data[0])
        print("...done")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

"""
Creates 2D Traveling Salesman Problems as a stack of coordinates uniformly
sampled in range [0, 1] for both dimensions.

Adapted from code provided with the paper "Attention, Learn to Solve Routing Problems!"
https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/problem_tsp.py
"""

import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

SENTINEL = -1.0


def gen_problem(prob_size, max_prob_size):
    row = torch.FloatTensor(max_prob_size, 2).uniform_(0, 1)

    if prob_size < max_prob_size:
        row[prob_size - max_prob_size :] = SENTINEL

    return row


class TspDataset(Dataset):
    """
    Load or generate TSP problems randomly sampled in [0, 1] ** 2.
    Specify 'filename' to load from a pickled list of FloatTensors.
    Otherwise, specify 'size' and 'num_samples' to randomly sample.

    If 'size' is an integer, all problems will be generated with this
    number of nodes. If 'size' is an iterator, problem sizes will be
    uniformly sampled from each provided size.
    """

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
            # determine each sample's problem size (constant if size is int)
            size_choices = (size,) if type(size) is int else tuple(size)
            sample_sizes = np.random.choice(size_choices, size=num_samples)

            # sample points randomly in [0, 1] square
            self.data = []
            max_problem_size = max(size_choices)

            for prob_size in sample_sizes:
                row = gen_problem(prob_size, max_problem_size)
                self.data.append(row)

        self.problem_sizes = size_choices
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TspLiveDatagen(Dataset):
    """
    Generate TSP problems on the fly.
    """

    def __init__(self, size=50, epoch_size=int(1e6)):
        """
        Specify problem sizes and a pseudo number of samples.
        The latter determines how many samples are in one virtual
        'epoch' to provide an API which conforms with DataLoader's
        expectations.
        """
        super(TspLiveDatagen, self).__init__()
        self.problem_sizes = (size,) if type(size) is int else tuple(size)
        self.max_prob_size = max(self.problem_sizes)
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        prob_size = np.random.choice(self.problem_sizes)
        return gen_problem(prob_size, self.max_prob_size)


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


class OracleDataset(Dataset):
    """
    Loads a supervised learning dataset, where the targets are ground truth
    examples generated by pyconcorde.

    Useful for imitation learning.
    """

    def __init__(self, data_path, label_path, seq_shuffle=True):
        """
        Will shuffle the problems and labels correspondingly
        along the sequence dimension when 'seq_shuffle' is
        asserted.

        WARNING: 'seq_shuffle' is only explicitly supported
            for fixed problem size datasets without padding.
            The rest of the pipeline might actually work with
            shuffled sentinels, but this has not been tested.
        """
        data = np.load(data_path)
        labels = np.load(label_path)

        b, n, _ = data.shape

        if seq_shuffle:
            from tsp.utils import perm_shuffle

            perms = np.stack([np.random.permutation(n) for _ in range(b)], axis=0)
            data, labels = perm_shuffle(data, labels, perms)

        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
        self.size = b

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])

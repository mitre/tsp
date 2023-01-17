import unittest
import torch
import torch.nn.functional as F

from tsp.model.submodules import TspEncoder, PositionalEncoding
from tsp.model.monty_style import TspMontyStyleDecoder, TspCritic
from tsp.utils import get_coords


def make_encodings(coords, dim_model, num_layers):
    encoder = TspEncoder(dim_model, num_layers)
    return encoder(coords)


class TestModel(unittest.TestCase):
    def test_encoder(self):
        batch_size = 32
        dim_model = 128

        coords = get_coords(batch_size, problem_size=10)

        self.assertEqual(coords.shape, torch.empty(batch_size, 10, 2).shape)

        encodings = make_encodings(coords, dim_model, 1)
        self.assertEqual(encodings.shape, torch.empty(batch_size, 10, dim_model).shape)

        encodings = make_encodings(coords, dim_model, 2)
        self.assertEqual(encodings.shape, torch.empty(batch_size, 10, dim_model).shape)

    def test_positional_encoding(self):
        batch_size = 32
        dim_model = 128
        problem_size = 10

        pencoder = PositionalEncoding(dim_model, problem_size)
        inputs = torch.rand(problem_size, batch_size, dim_model)

        pos_encs = pencoder(inputs)
        self.assertEqual(
            pos_encs.shape, torch.empty(problem_size, batch_size, dim_model).shape
        )

    def test_critic(self):
        batch_size = 32  # N
        dim_model = 128  # d
        problem_size = 10  # S

        selections = get_coords(batch_size, problem_size)
        node_encodings = torch.rand(batch_size, problem_size, dim_model)

        critic = TspCritic(dim_model, num_layers=2)
        values = critic(node_encodings, selections)

        self.assertEqual(values.shape, torch.empty(batch_size, problem_size).shape)

    def test_decoder_montystyle(self):
        batch_size = 32  # N
        dim_model = 128  # d
        problem_size = 10  # S

        coords = get_coords(batch_size, problem_size)
        node_encodings = make_encodings(coords, dim_model, 1)

        decoder = TspMontyStyleDecoder(dim_model, num_layers=2)

        selections = torch.zeros(batch_size, 1, 2)
        # select_pad_mask = torch.ones(selections.shape, dtype=torch.bool)
        output_mask = torch.zeros((batch_size, problem_size), dtype=torch.bool)

        for sel_idx in range(problem_size):
            log_probs = decoder(node_encodings, selections, output_mask)  # (N, S)

            self.assertTrue(
                torch.allclose(torch.exp(log_probs).sum(dim=1), torch.ones(batch_size))
            )

            next_idx = torch.argmax(log_probs, dim=1)  # greedy selection, shape (N,)
            next_idx_g = torch.stack([next_idx] * 2, dim=-1).unsqueeze(1)
            next_sel = torch.gather(coords, dim=1, index=next_idx_g)  # (N, 1, d)

            self.assertEqual(next_sel.shape, torch.empty(batch_size, 1, 2).shape)
            for bidx in range(batch_size):
                self.assertTrue(
                    torch.allclose(next_sel[bidx], coords[bidx, next_idx[bidx]])
                )

            selections = torch.cat((selections, next_sel), dim=1)
            output_mask = torch.logical_or(
                output_mask, F.one_hot(next_idx, num_classes=problem_size)
            )

            self.assertTrue(
                torch.allclose(  # check each mask slice in batch has sel_idx + 1 number of assertions
                    output_mask.sum(dim=1).to(torch.float),
                    torch.full((batch_size,), sel_idx + 1, dtype=torch.float),
                )
            )

        self.assertEqual(
            selections.shape, torch.empty(batch_size, problem_size + 1, 2).shape
        )  # +1 from start token
        self.assertTrue(torch.all(output_mask))

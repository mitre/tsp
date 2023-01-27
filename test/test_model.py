import unittest
from numpy import select
import torch
import torch.nn.functional as F

from tsp.model.submodules import TspEncoder, PositionalEncoding
from tsp.model.monty_style import TspMontyStyleDecoder, TspCritic
from tsp.utils import get_coords, generate_padding_mask, reset_pads
from tsp.datagen import SENTINEL


def make_encodings(coords, dim_model, num_layers):
    encoder = TspEncoder(dim_model, num_layers)
    return encoder(coords)


class TestSubmodules(unittest.TestCase):
    def test_encoder(self):
        batch_size = 32
        dim_model = 128

        coords = get_coords(batch_size, problem_size=10)

        self.assertEqual(coords.shape, torch.empty(batch_size, 10, 2).shape)

        encodings = make_encodings(coords, dim_model, 1)
        self.assertEqual(encodings.shape, torch.empty(batch_size, 10, dim_model).shape)

        encodings = make_encodings(coords, dim_model, 2)
        self.assertEqual(encodings.shape, torch.empty(batch_size, 10, dim_model).shape)

    def test_encoder_EOS_pad_masking(self):
        batch_size = 32
        dim_model = 128
        problem_size = range(5, 11)  # 5-10

        coords = get_coords(batch_size, problem_size)

        encodings = make_encodings(coords, dim_model, 1)
        self.assertEqual(encodings.shape, torch.empty(batch_size, 10, dim_model).shape)

        sntnl_b, sntnl_s = torch.nonzero((coords == SENTINEL)[:, :, 0], as_tuple=True)
        self.assertTrue(
            torch.allclose(
                encodings[sntnl_b, sntnl_s],
                torch.full((len(sntnl_b), dim_model), SENTINEL, dtype=torch.float),
            )
        )

        # note we can only assume key padding is done correctly here

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

        selections = get_coords(batch_size, problem_size + 1)
        node_encodings = torch.rand(batch_size, problem_size, dim_model)

        select_pad_mask = generate_padding_mask(selections)
        node_enc_pad_mask = torch.rand(batch_size, problem_size) < 0.1

        critic = TspCritic(dim_model, num_layers=2)
        values = critic(node_encodings, selections, node_enc_pad_mask, select_pad_mask)

        self.assertEqual(values.shape, torch.empty(batch_size, problem_size + 1).shape)

    def test_critic_EOS_pad_masking(self):
        batch_size = 32  # N
        dim_model = 128  # d
        problem_size = range(4, 7)  # S

        selections = get_coords(
            batch_size, range(min(problem_size) + 1, max(problem_size) + 2)
        )
        node_encodings = torch.rand(batch_size, max(problem_size), dim_model)

        select_pad_mask = generate_padding_mask(selections)
        node_enc_pad_mask = torch.rand(batch_size, max(problem_size)) < 0.1

        critic = TspCritic(dim_model, num_layers=2)
        values = critic(node_encodings, selections, node_enc_pad_mask, select_pad_mask)

        self.assertEqual(
            values.shape, torch.empty(batch_size, max(problem_size) + 1).shape
        )

        sntnl_b, sntnl_s = torch.nonzero(
            (selections == SENTINEL)[:, :, 0], as_tuple=True
        )
        self.assertTrue(
            torch.allclose(
                values[sntnl_b, sntnl_s],
                torch.full((len(sntnl_b),), SENTINEL, dtype=torch.float),
            )
        )


class TestMontyModel(unittest.TestCase):
    def _run_checks_decoder_montystyle(self, batch_size, dim_model, problem_size):
        coords = get_coords(batch_size, problem_size)
        node_encodings = make_encodings(coords, dim_model, 1)
        pad_mask = generate_padding_mask(coords)

        decoder = TspMontyStyleDecoder(dim_model, num_layers=2)

        max_problem_size = (
            problem_size if type(problem_size) is int else max(problem_size)
        )

        selections = torch.zeros(batch_size, 1, 2)
        select_idxs = torch.empty((batch_size, max_problem_size), dtype=torch.int64)
        log_probs = torch.empty(batch_size, max_problem_size, max_problem_size)
        output_mask = torch.zeros((batch_size, max_problem_size), dtype=torch.bool)

        for step_idx in range(max_problem_size):
            joint_invalid_mask = torch.logical_or(output_mask, pad_mask)
            eos_mask = torch.all(joint_invalid_mask, dim=1)

            log_probs[:, step_idx] = decoder(
                node_encodings, selections, output_mask, pad_mask, eos_mask
            )

            log_prob_pad_mask = pad_mask[:, step_idx]
            num_tours_undecided = (~log_prob_pad_mask).sum()

            self.assertTrue(
                torch.allclose(
                    torch.exp(log_probs[:, step_idx][~log_prob_pad_mask]).sum(dim=-1),
                    torch.ones(num_tours_undecided),
                )
            )

            next_idx = torch.argmax(
                log_probs[:, step_idx, :], dim=-1
            )  # greedy selection, shape (N,)
            next_idx_g = torch.stack([next_idx] * 2, dim=-1).unsqueeze(1)
            next_sel = torch.gather(coords, dim=1, index=next_idx_g)  # (N, 1, d)

            self.assertEqual(next_sel.shape, torch.empty(batch_size, 1, 2).shape)
            for bidx in range(batch_size):
                self.assertTrue(
                    torch.allclose(next_sel[bidx], coords[bidx, next_idx[bidx]])
                )

            selections = torch.cat((selections, next_sel), dim=1)
            output_mask[~log_prob_pad_mask] = torch.logical_or(
                output_mask[~log_prob_pad_mask],
                F.one_hot(next_idx[~log_prob_pad_mask], num_classes=max_problem_size),
            )

            self.assertTrue(
                torch.allclose(  # check each mask slice in batch has step_idx + 1 number of assertions
                    output_mask[~log_prob_pad_mask].sum(dim=1).to(torch.float),
                    torch.full((num_tours_undecided,), step_idx + 1, dtype=torch.float),
                )
            )

            select_idxs[:, step_idx] = next_idx

        self.assertEqual(
            selections.shape, torch.empty(batch_size, max_problem_size + 1, 2).shape
        )  # +1 from start token

        self.assertTrue(torch.all(torch.logical_or(output_mask, pad_mask)))
        self.assertFalse(torch.any(torch.logical_and(output_mask, pad_mask)))

        return coords, selections, select_idxs, log_probs, output_mask

    def test_decoder_montystyle(self):
        batch_size = 32  # N
        dim_model = 128  # d
        problem_size = 5  # S

        self._run_checks_decoder_montystyle(batch_size, dim_model, problem_size)

    def test_decoder_ms_EOS_pad_masking(self):
        batch_size = 32  # N
        dim_model = 128  # d
        problem_size = range(4, 7)  # S

        (
            coords,
            selections,
            select_idxs,
            log_probs,
            output_mask,
        ) = self._run_checks_decoder_montystyle(batch_size, dim_model, problem_size)

        self.assertFalse(torch.any(torch.isnan(log_probs)))

        selections = selections[:, 1:]

        pad_mask = generate_padding_mask(coords)
        selections = reset_pads(selections, pad_mask, val=SENTINEL)
        select_idxs = reset_pads(select_idxs, pad_mask, val=SENTINEL)

        sntnl_b, sntnl_s = torch.nonzero((coords == SENTINEL)[:, :, 0], as_tuple=True)
        self.assertTrue(
            torch.allclose(
                selections[sntnl_b, sntnl_s],
                torch.full((len(sntnl_b), 2), SENTINEL, dtype=torch.float),
            )
        )
        self.assertTrue(
            torch.allclose(
                select_idxs[sntnl_b, sntnl_s],
                torch.full((len(sntnl_b),), SENTINEL, dtype=torch.long),
            )
        )


if __name__ == "__main__":
    unittest.main()

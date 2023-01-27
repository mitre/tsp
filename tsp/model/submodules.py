from tsp.datagen import SENTINEL
import torch
import torch.nn as nn
import math

from tsp.utils import generate_square_subsequent_mask, generate_padding_mask, reset_pads


class TspEncoder(nn.Module):
    """
    Shared Transformer encoder module.
    Performs self-attention over all
    coordinates in a given problem.
    """

    def __init__(self, dim_model=128, num_layers=3):
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embed = nn.Linear(2, dim_model)  # hard-coded to expect 2D TspDataset

        encoder_layer = nn.TransformerEncoderLayer(
            dim_model, nhead=8, dim_feedforward=(4 * dim_model), dropout=0.0
        )

        self.trf_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, problems):
        """
        Assumes 'problems' has shape (N, S, 2)
            N = batch dim
            S = seq dim
        """
        pad_mask = generate_padding_mask(problems)

        problems_t = torch.transpose(problems, 0, 1)
        symmetric_input = problems_t * 2 - 1  # [0, 1] --> [-1, 1]
        enc_input = self.embed(symmetric_input)

        out_t = self.trf_enc(enc_input, src_key_padding_mask=pad_mask)
        out = torch.transpose(out_t, 0, 1)

        return reset_pads(out, pad_mask, val=SENTINEL)


class PositionalEncoding(nn.Module):
    """
    Adapted from PyTorch tutorials:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Encoding addends are in [-1, 1].

    NOTE: Unlike TspEncoder and TspDecoder modules,
    this module assumes data is already shaped (S, N, 2),
    where the sequence dim comes **before** the batch dim,
    in contrast to the (N, S, 2) shape seen elsewhere.
    (Presumably this module is used inside the encoder or
    decoder, where this transpose already needs to occur).
    """

    def __init__(self, d_model, max_len=100, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TspCritic(nn.Module):
    """
    TODO
    """

    def __init__(self, dim_model=128, num_layers=2):
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embed = nn.Linear(2, dim_model)  # hard-coded to expect 2D TspDataset
        self.pos_enc = PositionalEncoding(dim_model)

        decoder_layer = nn.TransformerDecoderLayer(
            dim_model, nhead=8, dim_feedforward=(4 * dim_model), dropout=0.0
        )

        self.trf_dec = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.out = nn.Sequential(
            nn.Linear(dim_model, dim_model), nn.ReLU(), nn.Linear(dim_model, 1)
        )

    def forward(
        self, node_encodings, selections, node_enc_padding_mask, select_padding_mask
    ):
        """
        Note 'selections' must include a start
        token to provide problem value estimates
        not conditioned on actor decisions.
        """
        node_encodings_t = torch.transpose(node_encodings, 0, 1)
        selections_t = torch.transpose(selections, 0, 1)

        sel_input_sym = selections_t * 2 - 1  # [0, 1] --> [-1, 1]
        sel_input_emb = self.embed(sel_input_sym)
        sel_input_pe = self.pos_enc(sel_input_emb)

        # decoder step with language-model-style tgt and mem (not flipped like in TspMontyStyleDecoder)
        self_attn_mask = generate_square_subsequent_mask(len(sel_input_pe)).to(
            sel_input_pe.device
        )
        dec_out = self.trf_dec(
            sel_input_pe,
            node_encodings_t,
            tgt_mask=self_attn_mask,
            tgt_key_padding_mask=select_padding_mask,
            memory_key_padding_mask=node_enc_padding_mask,
        )

        vals_t = self.out(dec_out).squeeze(-1)
        vals = torch.transpose(vals_t, 0, 1)

        return reset_pads(vals, select_padding_mask, val=SENTINEL)

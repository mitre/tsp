import torch
import torch.nn as nn
import torch.nn.functional as F

from tsp.model.submodules import TspEncoder, PositionalEncoding, TspCritic


class TspMontyStyleDecoder(nn.Module):
    """
    Node encodings and selections switch places
    as target and memory, respectively.

    Self-attention is applied to the node
    encodings like in the TspEncoder.

    For the encoder-decoder attention step,
    node encodings are the queries and
    the selected coordinates thus far are
    used to generate keys and values.

    This results in a decoder-stack output shape
    of (S, N, 2), or (seq dim, batch dim, 2).

    A linear layer then condenses the coordinate
    embedding dim '2' and we perform a masked softmax
    over the sequence dim 'S' (number of nodes in TSP
    problem).
    """

    def __init__(self, dim_model=128, num_layers=3):
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embed = nn.Linear(2, dim_model)  # hard-coded to expect 2D TspDataset
        self.pos_enc = PositionalEncoding(dim_model)

        self.to_dist = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, 1),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            dim_model, nhead=8, dim_feedforward=(4 * dim_model), dropout=0.0
        )

        self.trf_dec = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

    def forward(self, node_encodings, selections, output_mask):
        """
        Generate selection distribution for the next node using
        self-attention over node encodings and encoder-decoder
        attention over the coordinates selected so far.

        Args:
            node_encodings - encoded rep of problem coordinates, shape (N, S, 2)
            selections - ordered problem coordinates selected so far, shape (N, ?, 2)
            output_mask - boolean mask asserted at indices already selected, shape (N, S)

        Returns log_probs over next selection, shape (N, S).

        Note '?' is a wildcard of selections. For the first iteration of decoding
        this should just be size 1 for the start token.

        TODO For now omitting 'select_pad_mask' which may serve as
        the memory_key_padding_mask when using batches of different
        TSP problem sizes (where number of selections varies).
        """
        node_encodings_t = torch.transpose(node_encodings, 0, 1)
        selections_t = torch.transpose(selections, 0, 1)
        output_mask_t = torch.transpose(output_mask, 0, 1)

        sel_input_sym = selections_t * 2 - 1  # [0, 1] --> [-1, 1]
        sel_input_emb = self.embed(sel_input_sym)
        sel_input_pe = self.pos_enc(sel_input_emb)

        dec_out = self.trf_dec(node_encodings_t, sel_input_pe)

        logits = self.to_dist(dec_out).squeeze(-1)
        logits[output_mask_t] = float("-inf")  # make re-selection impossible

        log_probs = F.log_softmax(logits, dim=0)
        return torch.transpose(log_probs, 0, 1)


class TspMontyStyleModel(nn.Module):
    def __init__(self, dim_model=128, num_enc_layers=3, num_dec_layers=3):
        super().__init__()
        self.dim_model = dim_model
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        self.encoder = TspEncoder(self.dim_model, num_layers=num_enc_layers)
        self.decoder = TspMontyStyleDecoder(self.dim_model, num_layers=num_dec_layers)

    def forward(self, problems, select_fn):
        selections, select_idxs, log_probs, _, _ = self._rollout(problems, select_fn)
        selections = selections[:, 1:]  # remove start token before returning

        return selections, select_idxs, log_probs

    def _rollout(self, problems, select_fn):
        """
        TODO
        """
        batch_size, problem_size, _ = problems.shape
        node_encodings = self.encoder(problems)

        selections = torch.zeros(batch_size, 1, 2).to(
            problems.device
        )  # zeros as start token
        select_idxs = torch.empty((batch_size, problem_size), dtype=torch.int64).to(
            problems.device
        )
        log_probs = torch.empty(batch_size, problem_size, problem_size).to(
            problems.device
        )
        output_mask = torch.zeros((batch_size, problem_size), dtype=torch.bool).to(
            problems.device
        )

        for step_idx in range(problem_size):
            selections, step_sel_idx, step_log_probs, output_mask = self.step(
                problems, node_encodings, selections, output_mask, select_fn
            )

            select_idxs[:, step_idx] = step_sel_idx  # save integer index of last choice
            log_probs[
                :, step_idx
            ] = step_log_probs  # save log prob dist from last choice

        return selections, select_idxs, log_probs, node_encodings, output_mask

    def step(self, problems, node_encodings, selections, output_mask, select_fn):
        """
        TODO

        Detach next_sel --> don't want grads to propagate through past selections
        Though I think this is unnecessary since it depends on the problems and
        index tensors, both of which are usually leaf tensors (for the index tensor,
        because we use argmax or a sample).
        """
        problem_size = problems.shape[1]

        log_probs = self.decoder(node_encodings, selections, output_mask)
        next_idx, next_sel = select_fn(problems, log_probs)

        selections = torch.cat((selections, next_sel.detach()), dim=1)
        output_mask = torch.logical_or(
            output_mask, F.one_hot(next_idx, num_classes=problem_size)
        )

        return selections, next_idx, log_probs, output_mask


class TspMsAcModel(TspMontyStyleModel):
    """
    TSP Monty-style Actor-crtic model.
    Simply adds a value head over selections.
    """

    def __init__(
        self, dim_model=128, num_enc_layers=3, num_dec_layers=3, num_crt_layers=3
    ):
        """Initialize actor model and decoupled critic model."""
        super().__init__(
            dim_model=dim_model,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
        )
        self.num_crt_layers = num_crt_layers

        self.critic = TspCritic(self.dim_model, num_layers=num_crt_layers)

    def forward(self, problems, select_fn):
        """
        Call TspMontyStyleModel to get solections and log probs.
        Then compute values for each selection sequence; values
        are based on all selections up to their sequence index
        (inclusive).
        """
        selections, select_idxs, log_probs, node_encodings, _ = self._rollout(
            problems, select_fn
        )
        values = self.critic(
            node_encodings, selections
        )  # selections include start token

        selections = selections[:, 1:]  # remove start token before returning

        return selections, select_idxs, log_probs, values

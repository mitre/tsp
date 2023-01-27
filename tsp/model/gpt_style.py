
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsp.model.submodules import PositionalEncoding, SingleHeadAttnPtr
from tsp.utils import bool_to_additive_mask, generate_square_subsequent_mask


def _make_index_tensor(value):
    inds = torch.arange(value.shape[1])
    inds = inds.view(1, -1).repeat(value.shape[0], 1)
    return inds


def before_start_token_mask(value, token, include_token=False):
    """
    Creates a mask for all of the input tokens before the start token

    value: (B, S, N)
    token: (N)

    output: (B, S)
    """
    inds = _make_index_tensor(value)
    token_locations = torch.all(torch.eq(value, token), dim=-1)
    token_locations = inds[token_locations]

    return inds < token_locations.view(-1, 1) + include_token


def attention_mask(value):
    """
    Makes the standard attention mask
    value: (B, S, N)

    mask: (S, S)
    """
    b, s, n = value.shape
    inds = _make_index_tensor(torch.ones(s, s))
    mask = inds <= inds.T
    return mask

class GPTDecoder(nn.Module):

    def __init__(self, dim_model, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.dim_model = dim_model

        self.first = nn.Linear(2, self.dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            dim_model, nhead=8, dim_feedforward=(4 * dim_model), dropout=0.0
        )

        self.trf = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.mha = nn.MultiheadAttention(self.dim_model, self.dim_model)

        self.start_token = -torch.ones(1, 1, 2)

    def append_start_token(self, tensor):
        repeated_token = self.start_token.repeat(tensor.shape[0], 1, 1)
        return torch.cat([tensor, repeated_token], dim=1)
    
    def sel_mask(self, problem):
        mask = before_start_token_mask(problem, self.start_token)
        # This assumes that the start token is in the same place for everyone
        expanded_mask = mask[0].unsqueeze(0)
        # print(expanded_mask)
        return expanded_mask.repeat(mask.shape[1], 1)

    def forward(self, problem):

        out_mask = torch.transpose(self.sel_mask(problem), 0, 1)
        
        mask_t = torch.transpose(before_start_token_mask(problem, self.start_token), 0, 1)

        problem_t = torch.transpose(problem, 0, 1)
        problem_t = self.first(problem_t)
        nodes_out_t = self.trf(problem_t)

        nodes_out = torch.transpose(nodes_out_t, 0, 1)
        
        out, logits = self.mha(nodes_out_t, nodes_out_t, nodes_out_t, attn_mask=~out_mask)

        out = torch.transpose(out, 0, 1)

        # print(out.shape)
        # print(logits.shape)
        # print(logits[0])

        #log_probs = F.log_softmax(logits, dim=0)

        #return log_probs


class TspGptPointerDecoder(nn.Module):
    """
    Decoder-only policy for TSP.
    
    Supports single backward pass RL training
    and single forward and backward pass supervised
    training.

    Node and selection embeddings are concatenated
    into the same sequence, separated by a start
    token. During masked self-attention, node 
    queries attend over each other, while 
    selections attend over previous 
    selections in addition to all nodes. 
    The output of the decoder stack has shape 
    (n + s + 1, N, 2) where n, s, N are the
    problem size, number of past selections, and
    batch size, respectively.

    Logits are generated using single-head attention
    weights of selection encodings attending over the 
    node encodings.
    """

    def __init__(self, dim_model=128, num_layers=6):
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embed = nn.Linear(2, dim_model)  # hard-coded to expect 2D TspDataset
        self.pos_enc = PositionalEncoding(dim_model)

        self.sha = SingleHeadAttnPtr(dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            dim_model, nhead=8, dim_feedforward=(4 * dim_model), dropout=0.0
        )

        self.trf = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def _embed_coords(self, coords):
        coords_sym = coords * 2 - 1  # [0, 1] --> [-1, 1]
        return self.embed(coords_sym)

    def _cat_input(self, node_input, sel_input):
        batch_size = node_input.shape[1]
        start_token = torch.zeros(1, batch_size, self.dim_model).to(node_input.device)
        return torch.cat((node_input, start_token, sel_input), dim=0)

    def _get_attn_mask(self, num_nodes, num_sel):
        src_size = num_nodes + num_sel + 1  # +1 for start token
        mask = torch.zeros(src_size, src_size)

        mask[:num_nodes, num_nodes:] = float("-inf")  # no node attention over selections
        mask[num_nodes:, num_nodes:] = generate_square_subsequent_mask(num_sel + 1)  # +1 for start token

        return mask        

    def forward(self, nodes, selections, output_mask):
        """
        TODO
        
        Args:
            nodes - non-ordered problem coordinates, shape (N, S, 2)
            selections - ordered problem coordinates selected so far, shape (N, s, 2)
            output_mask - boolean mask asserted at indices already selected, shape (N, s + 1, S)

        Where 's' is the length of the sequence so far, starting at s = 0.
        On the first decision, selections should be an empty array.
        Note this differs from TspMontyStyleDecoder, where the start
        token is provided by TspMontyStyleModel.
        """
        nodes_t = torch.transpose(nodes, 0, 1)
        num_nodes = nodes_t.shape[0]
        node_input = self._embed_coords(nodes_t)
        
        if len(selections) > 0:
            selections_t = torch.transpose(selections, 0, 1)
            num_sel = selections_t.shape[0]

            sel_input = self._embed_coords(selections_t)
            sel_input = self.pos_enc(sel_input)
        
        else:
            num_sel = 0
            sel_input = torch.empty(0).to(node_input.device)

        model_input = self._cat_input(node_input, sel_input)
        self_attn_mask = self._get_attn_mask(num_nodes, num_sel).to(model_input.device)

        trf_out = self.trf(model_input, mask=self_attn_mask)

        node_encs = trf_out[:num_nodes]
        sel_encs = trf_out[num_nodes:]  # includes initial rep from start token

        log_probs = self.sha(sel_encs, node_encs, bool_to_additive_mask(output_mask))  # (N, s+1, S)
        sel_encs = torch.transpose(sel_encs, 0, 1)  # (N, s+1, 2)
        
        return log_probs, sel_encs


class TspGptPointerModel(nn.Module):
    def __init__(self, dim_model=128, num_layers=6):
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers

        self.decoder = TspGptPointerDecoder(dim_model, num_layers=num_layers)

    def forward(self, problems, select_fn):
        selections, select_idxs, log_probs, _, _ = self._rollout(problems, select_fn)
        return selections, select_idxs, log_probs

    def _rollout(self, problems, select_fn):
        batch_size, problem_size, _ = problems.shape

        selections = torch.empty(0).to(
            problems.device
        )  # start token provided in decoder
        select_idxs = torch.empty((batch_size, problem_size), dtype=torch.int64).to(
            problems.device
        )
        output_mask = torch.zeros((batch_size, 1, problem_size), dtype=torch.bool).to(
            problems.device
        )

        ### TODO fix backprop bug creating nan grads when using final decoding steps sliced log_probs
        # don't need grad tracking until final action rollout
        with torch.no_grad():  
            for step_idx in range(problem_size - 1):
                selections, step_sel_idx, _, _, output_mask = self.step(
                    problems, selections, output_mask, select_fn
                )
                select_idxs[:, step_idx] = step_sel_idx  # save integer index of last choice
        
        # grad track final forward pass which contains all decisions plus last full-state selection encoding
        selections, step_sel_idx, log_probs, sel_encs, output_mask = self.step(
            problems, selections, output_mask, select_fn
        )
        select_idxs[:, -1] = step_sel_idx  # save integer index of last choice

        _, sel_encs = self.decoder(problems, selections, output_mask)
        # log_probs = log_probs[:, :-1]  # remove final nan distribution (nothing to select)
        ### END TODO

        return selections, select_idxs, log_probs, sel_encs, output_mask

    def step(self, problems, selections, output_mask, select_fn):
        problem_size = problems.shape[1]

        log_probs, sel_encs = self.decoder(problems, selections, output_mask)
        next_idx, next_sel = select_fn(problems, log_probs[:, -1])

        selections = torch.cat((selections, next_sel.detach()), dim=1)
        
        next_mask_slice = torch.logical_or(
            output_mask[:, -1], F.one_hot(next_idx, num_classes=problem_size)
        ).unsqueeze(1)
        output_mask = torch.cat((output_mask, next_mask_slice), dim=1)

        return selections, next_idx, log_probs, sel_encs, output_mask


class TspGptAcModel(TspGptPointerModel):
    def __init__(self, dim_model=128, num_dec_layers=6, value_head_layers=2):

        super().__init__(
            dim_model=dim_model,
            num_layers=num_dec_layers,
        )
        self.value_head_layers = value_head_layers

        self.critic = nn.Sequential(
            *([nn.Linear(dim_model, dim_model), nn.ReLU()] * value_head_layers),
            nn.Linear(dim_model, 1),
        )

    def forward(self, problems, select_fn):
        selections, select_idxs, log_probs, sel_encs, _ = self._rollout(
            problems, select_fn
        )
        
        values = self.critic(sel_encs).squeeze(-1)  # predict from selection encodings

        return selections, select_idxs, log_probs, values

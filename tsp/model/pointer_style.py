import torch
import torch.nn as nn


class TspPointerDecoder(nn.Module):
    """
    Follows decoding steps in vanilla Transformer.

    Self-attention is applied to the iteratively
    generated selected coordinates target sequence.

    For the encoder-decoder attention step,
    self-attention selection features are the
    queries and the node encodings produce the
    key/value pairs.

    This results in a decoder-stack output shape
    of (T, N, 2), or (target dim, batch dim, 2).

    A final single-head self-attention layer and
    softmax converts each target to log probs
    scoring next node selection, while masking out
    already  selected nodes. The output shape is
    (N, T, S), where 'S' is the encoder sequence
    dimension.
    """

    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

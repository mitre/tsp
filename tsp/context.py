from dataclasses import dataclass
from typing import Any, List, Tuple

from tsp.datagen import TspLiveDatagen, TspDataset


def load_eval_datasets(nodes, eval_samples):
    eval_datasets = [("all", TspDataset(size=nodes, num_samples=eval_samples))]

    if len(nodes) > 1:
        # also eval on data strictly containing lower and upper problem size bounds
        low, high = min(nodes), max(nodes)
        eval_datasets += [(f"{low}n", TspDataset(size=low, num_samples=eval_samples))]
        eval_datasets += [(f"{high}n", TspDataset(size=high, num_samples=eval_samples))]

    return eval_datasets


@dataclass(frozen=True)
class DataContext:
    nodes: Tuple[int] = None
    node_range: Tuple[int] = None
    is_eval: bool = False

    def __call__(self, total_samples):

        nodes = self.nodes
        if self.nodes is not None and self.node_range is not None:
            raise ValueError(
                "May specify custom '--nodes' or '--node_range' but not both"
            )
        elif self.node_range is not None:
            start, end = self.node_range
            nodes = range(start, end + 1)  # inclusive end bound
        elif self.nodes is None:
            nodes = (20,)  # default

        if not self.is_eval:
            return TspLiveDatagen(size=nodes, epoch_size=total_samples)
        else:
            return load_eval_datasets(nodes, total_samples)


@dataclass(frozen=True)
class ModelContext:
    model_type: Any
    model_dim: int
    n_enc: int
    n_dec: int

    def __call__(self, device, check_path=None):
        model = self.model_type(
            dim_model=self.model_dim,
            num_enc_layers=self.n_enc,
            num_dec_layers=self.n_dec,
        )

        if check_path:
            model.load_state_dict(torch.load(check_path, map_location=device))
        else:
            model = model.to(device)

        return model


@dataclass(frozen=True)
class AlgoContext:
    algo_type: Any
    optim_type: Any
    lr: float
    grad_norm_clip: float = 1.0

    def __call__(self, rank, agent, eval_set):
        optimizer = self.optim_type(agent.parameters(), lr=self.lr)

        algo = self.algo_type(
            rank, optimizer, eval_set, grad_norm_clip=self.grad_norm_clip
        )
        return algo

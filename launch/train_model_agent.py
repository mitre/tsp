import argparse

import torch
from math import ceil

from tsp.datagen import TspLiveDatagen, TspDataset
from tsp.model.monty_style import (
    TspMontyStyleModel,
    TspMsAcModel,
    TspMsGreedyBaselineModel,
)
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C, TspA2CGreedyRollout
from tsp.train import TspTrainer
from tsp.logger import get_latest_check_path
from tsp.utils import RangeMap

parser = argparse.ArgumentParser()

parser.add_argument(
    "name", nargs="?", default=None, type=str
)  # logs to <install_path>/runs/<name>/progress.csv (if not provided, no logging occurs)
parser.add_argument(
    "--nodes", nargs="+", default=None, type=int
)  # can be space-separated list of problem sizes
parser.add_argument(
    "--node_range", nargs=2, default=None, type=int
)  # use this or --nodes but not both
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--eval_samples", default=10000, type=int)
parser.add_argument("--eval_batch_size", default=1000, type=int)
parser.add_argument("--itr", default=250, type=float, help="in thousands")
parser.add_argument("--check_period", default=100, type=int)
parser.add_argument("--eval_period", default=1000, type=int)
parser.add_argument("--grad_norm_clip", default=1.0, type=float)
parser.add_argument("--critic_coeff", default=1.0, type=float)
parser.add_argument("--model_dim", default=128, type=int)
parser.add_argument("--n_enc", default=6, type=int)
parser.add_argument("--n_dec", default=6, type=int)
parser.add_argument("--n_crt", default=6, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--device", default=None, type=int)
parser.add_argument("--params", default=None)
parser.add_argument("--log_dir", default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--num_draw", default=0, type=int)  # per eval dataset
parser.add_argument("--draw_spec", nargs="+", default=None, type=int)
# --draw_spec W X Y Z == period W until itr X (thousands), then period Y until Z, etc.


def interpret_problem_sizes(args):
    if args.nodes is not None and args.node_range is not None:
        raise ValueError("May specify custom '--nodes' or '--node_range' but not both")
    elif args.node_range is not None:
        start, end = args.node_range
        return range(start, end + 1)  # inclusive end bound
    elif args.nodes is not None:
        return args.nodes
    else:
        return (20,)  # default


if __name__ == "__main__":

    args = parser.parse_args()

    nodes = interpret_problem_sizes(args)

    # create TSP datasets
    total_samples = ceil(1e3 * args.itr * args.batch_size)
    dataset = TspLiveDatagen(size=nodes, epoch_size=total_samples)

    eval_datasets = [("all", TspDataset(size=nodes, num_samples=args.eval_samples))]

    if len(nodes) > 1:
        # also eval on data strictly containing lower and upper problem size bounds
        low, high = min(nodes), max(nodes)
        eval_datasets += [
            (f"{low}n", TspDataset(size=low, num_samples=args.eval_samples))
        ]
        eval_datasets += [
            (f"{high}n", TspDataset(size=high, num_samples=args.eval_samples))
        ]

    # initialize model and agent
    model = TspMsGreedyBaselineModel(
        dim_model=args.model_dim,
        num_enc_layers=args.n_enc,
        num_dec_layers=args.n_dec,
    )
    agent = TspAgent(model, use_available_device=False)

    if args.device is not None:
        agent.to(args.device)
    else:
        args.device = "cpu"

    if args.resume and args.params is not None:
        raise ValueError("Provided checkpoint to load and --resume flag together are not supported")
    elif args.resume:
        check_path = get_latest_check_path(args.name, args.log_dir)
        agent.load_state_dict(
            torch.load(check_path, map_location=torch.device(args.device))
        )
    elif args.params is not None:
        agent.load_state_dict(
            torch.load(args.params, map_location=torch.device(args.device))
        )

    # initialize algorithm and optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    algo = TspA2CGreedyRollout(
        optimizer, eval_datasets[0][1], grad_norm_clip=args.grad_norm_clip
    )

    # construct draw period range map
    if args.draw_spec is not None:
        assert len(args.draw_spec) % 2 == 0, "Invalid draw spec format, see docs next to definition"
        partition = (0, *args.draw_spec[1::2])
        partition = [1000 * p for p in partition]
        values = (*args.draw_spec[::2],)
        draw_period_map = RangeMap(partition, values)
    else:
        draw_period_map = None

    # build runner and start training
    runner = TspTrainer(
        dataset,
        agent,
        algo,
        args.name,
        eval_datasets=eval_datasets,
        num_tours_draw=args.num_draw,
        log_dir=args.log_dir,
        resume=args.resume
    )

    runner.start(
        epochs=1,
        batch_size=args.batch_size,
        check_period=args.check_period,
        eval_period=args.eval_period,
        eval_batch_size=args.eval_batch_size,
        draw_period_map=draw_period_map
    )

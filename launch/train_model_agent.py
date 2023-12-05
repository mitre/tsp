import argparse

import torch
from math import ceil

from tsp.datagen import TspLiveDatagen, TspDataset
from tsp.model.monty_style import TspMontyStyleModel, TspMsAcModel
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C, TspPPO
from tsp.train import TspTrainer

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
parser.add_argument("--algo", default="ppo", choices=["a2c", "ppo"], type=str, help="train with compositional A2C or PPO; defaults to PPO")
parser.add_argument("--minibatch_epochs", default=1, type=int, help="PPO only; number of minibatch epochs aka how many times to use data before getting new samples")
parser.add_argument("--minibatches", default=1, type=int, help="PPO only; how many minibatches (gradient updates) to take per minibatch_epoch")
parser.add_argument("--ratio_clip", default=0.1, type=float, help="PPO only; clamp threshold during creation of surrogate objectives in loss (smaller means more stable but more constrained gradient updates)")


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
    model = TspMsAcModel(
        dim_model=args.model_dim,
        num_enc_layers=args.n_enc,
        num_dec_layers=args.n_dec,
        num_crt_layers=args.n_crt,
    )
    agent = TspAgent(model)

    if args.device is not None:
        agent.to(args.device)

    # initialize algorithm and optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    if args.algo == "ppo":
        AlgoCls = TspPPO
        extra_algo_kwargs = dict(
            epochs=args.minibatch_epochs,
            minibatches=args.minibatches,
            ratio_clip=args.ratio_clip
        )
    elif args.algo == "a2c":
        AlgoCls = TspA2C
        extra_algo_kwargs = dict()  # no extra kwargs
    else:
        raise Exception(f"Unrecognized training algorithm '{args.algo}'")

    algo = AlgoCls(
        optimizer, grad_norm_clip=args.grad_norm_clip, critic_coeff=args.critic_coeff, **extra_algo_kwargs
    )

    # build runner and start training
    runner = TspTrainer(dataset, agent, algo, args.name, eval_datasets=eval_datasets)

    runner.start(
        epochs=1,
        batch_size=args.batch_size,
        check_period=args.check_period,
        eval_period=args.eval_period,
        eval_batch_size=args.eval_batch_size,
    )

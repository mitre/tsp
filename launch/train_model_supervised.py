import argparse

import torch

from tsp.datagen import TspDataset, OracleDataset
from tsp.model.monty_style import (
    TspMontyStyleModel,
    TspMsAcModel,
    TspMsGreedyBaselineModel,
)
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspActorCriticSupervised, TspSupervisedGreedyRollout
from tsp.train import TspTrainer
from tsp.logger import get_latest_check_path

parser = argparse.ArgumentParser()

parser.add_argument(
    "name", nargs="?", default=None, type=str
)  # logs to <install_path>/runs/<name>/progress.csv (if not provided, no logging occurs)
parser.add_argument("--nodes", default=20, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--eval_batch_size", default=1000, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--check_period", default=100, type=int)
parser.add_argument("--eval_period", default=100, type=int)
parser.add_argument("--grad_norm_clip", default=1.0, type=float)
parser.add_argument("--critic_coeff", default=1.0, type=float)
parser.add_argument("--model_dim", default=128, type=int)
parser.add_argument("--n_enc", default=6, type=int)
parser.add_argument("--n_dec", default=6, type=int)
parser.add_argument("--n_crt", default=6, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--device", default=None, type=int)
parser.add_argument("--no_seq_shuffle", action="store_true", default=False)
parser.add_argument("--params", default=None)
parser.add_argument("--keep_check", action="store_true", default=False)
parser.add_argument("--log_dir", default=None)
parser.add_argument("--resume", action="store_true", default=False)


if __name__ == "__main__":

    args = parser.parse_args()

    # create TSP datasets
    dataset = OracleDataset(
        "example_shuffle_data.npy",
        "example_shuffle_labels.npy",
        seq_shuffle=(not args.no_seq_shuffle),
    )
    eval_dataset = [("all", TspDataset(size=args.nodes, num_samples=10000))]

    # initialize model and agent
    model = TspMsGreedyBaselineModel(
        dim_model=args.model_dim,
        num_enc_layers=args.n_enc,
        num_dec_layers=args.n_dec,
    )
    agent = TspAgent(model)

    if args.device is not None:
        agent.to(args.device)
    else:
        args.device = "cpu"

    if args.resume and args.params is not None:
        raise ValueError(
            "Provided checkpoint to load and --resume flag together are not supported"
        )
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
    algo = TspSupervisedGreedyRollout(optimizer, grad_norm_clip=args.grad_norm_clip)

    # build runner and start training
    runner = TspTrainer(
        dataset,
        agent,
        algo,
        args.name,
        eval_datasets=eval_dataset,
        log_dir=args.log_dir,
        resume=args.resume,
    )

    runner.start(
        epochs=args.epochs,
        batch_size=args.batch_size,
        check_period=args.check_period,
        eval_period=args.eval_period,
        eval_batch_size=args.eval_batch_size,
        keep_all_check=args.keep_check,
    )

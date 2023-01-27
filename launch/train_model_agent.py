import argparse
import time

from tsp.model.monty_style import TspMsGreedyBaselineModel
from tsp.train import TrainingContext
from tsp.context import DataContext, ModelContext, AlgoContext
from tsp.algo import TspA2CGreedyRollout

import torch

import torch.multiprocessing as mp

parser = argparse.ArgumentParser()

# logs to <install_path>/runs/<name>/progress.csv (if not provided, no logging occurs)
parser.add_argument("name", nargs="?", default=None, type=str)

# use this or --nodes but not both
parser.add_argument("--nodes", nargs="+", default=None, type=int)

# can be space-separated list of problem sizes
parser.add_argument("--node_range", nargs=2, default=None, type=int)

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
parser.add_argument("--keep_check", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
# world size
parser.add_argument("--np", default=1, type=int)
parser.add_argument("--rank", nargs="+", default=None, type=int)
parser.add_argument("--gpu", default=None, type=int)

if __name__ == "__main__":

    args = parser.parse_args()

    train_data = DataContext(args.nodes, args.node_range, is_eval=False)
    eval_data = DataContext(args.nodes, args.node_range, is_eval=True)
    model_context = ModelContext(
        TspMsGreedyBaselineModel, args.model_dim, args.n_enc, args.n_dec
    )
    algo_context = AlgoContext(
        TspA2CGreedyRollout, torch.optim.Adam, args.lr, args.grad_norm_clip
    )
    training_context = TrainingContext(
        args.name,
        args.batch_size,
        args.eval_samples,
        args.eval_batch_size,
        args.itr,
        args.check_period,
        args.eval_period,
        args.log_dir,
        args.keep_check,
        args.resume,
    )

    train_fn = training_context(model_context, algo_context, train_data, eval_data)

    if isinstance(args.rank, list):

        if len(args.rank) == 1:
            rank = args.rank[0]

    if args.np == 1:
        train_fn(0, args.np, gpu=args.gpu)
    if args.np > 1 and args.rank is None:
        mp.spawn(train_fn, args=(args.np, args.gpu), nprocs=args.np)
    else:
        processes = []
        for r in args.rank:
            p = mp.Process(target=train_fn, args=(r, args.np, args.gpu))
            p.start()

        for p in processes:
            p.join()

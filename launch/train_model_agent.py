import argparse

import torch

from tsp.datagen import TspDataset
from tsp.model.monty_style import TspMontyStyleModel, TspMsAcModel
from tsp.model.gpt_style import TspGptPointerModel, TspGptAcModel
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C
from tsp.train import TspTrainer

parser = argparse.ArgumentParser()

parser.add_argument("name", nargs='?', default=None, type=str)  # logs to <install_path>/runs/<name>/progress.csv (if not provided, no logging occurs)
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


if __name__ == "__main__":

    args = parser.parse_args()

    # create TSP datasets
    dataset = TspDataset(size=args.nodes, num_samples=12800000)
    eval_dataset = TspDataset(size=args.nodes, num_samples=10000)

    # initialize model and agent
    model = TspGptAcModel(
        dim_model=args.model_dim,
        num_dec_layers=6,
        value_head_layers=2
    )
    agent = TspAgent(model)

    if args.device is not None:
        agent.to(args.device)

    # initialize algorithm and optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    algo = TspA2C(optimizer, grad_norm_clip=args.grad_norm_clip, critic_coeff=args.critic_coeff)

    # build runner and start training
    runner = TspTrainer(dataset, agent, algo, args.name, eval_dataset=eval_dataset)

    runner.start(
        epochs=args.epochs,
        batch_size=args.batch_size,
        check_period=args.check_period,
        eval_period=args.eval_period,
        eval_batch_size=args.eval_batch_size,
    )

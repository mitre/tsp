from time import time
import os
from math import ceil
from dataclasses import dataclass
import os.path as osp
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from tsp.logger import Logger, get_log_dir
from tsp.datagen import TspDataset
from tsp.eval import batched_eval
from tsp.agent import TspAgent


def train(
    iteration,
    dataloader,
    agent,
    algo,
    eval_fn,
    log_fn,
    epochs=1,
):

    for epoch_idx in range(epochs):

        for batch_idx, problem_batch in tqdm(enumerate(dataloader)):
            # train routine
            agent.train_mode()
            algo.optimize_agent(iteration, agent, problem_batch)

            # (offline) eval routine
            eval_fn(agent, iteration)
            log_fn(agent, iteration)

            iteration += 1


def log_step(
    rank, start_time, time_offset, keep_all_check, check_period, agent, iteration
):
    """
    log writing, checkpoint saving, etc.
    """
    Logger.log("iteration", iteration)
    Logger.log("time", time() - start_time + time_offset)
    Logger.dump()

    if iteration % check_period == 0 and rank == 0:
        Logger.checkpoint(agent.state_dict(), iteration, keep_prev=keep_all_check)


def setup_logging(rank, run_name, log_dir, resume):
    """
    Sets up the logging object, returns current run state in the form of the
    last iteration and time
    """
    last_itr = 0
    last_time = 0
    if run_name is not None:

        if log_dir is None:
            log_dir = get_log_dir(run_name)

        output_path = osp.join(log_dir, run_name)
        last_itr, last_time = Logger.init(rank, output_path, resume=resume)

    else:
        Logger.dummy_init()

    return last_itr, last_time


def offline_eval_routine(eval_data, eval_period, eval_batch_size, agent, iteration):
    """
    Run offline evaluation using all provided eval datasets.
    """
    agent.eval_mode()
    if eval_period is not None and iteration % eval_period == 0:
        for name, problems in eval_data:
            eval_costs = batched_eval(agent, problems, batch_size=eval_batch_size)
            Logger.log_stat(f"eval_cost_{name}", eval_costs)
    elif eval_period is not None:
        for name, _ in eval_data:
            Logger.log_stat(f"eval_cost_{name}", None)  # null value to satisfy logger


def parse_eval_datasets(eval_datasets):
    """
    Converts the input eval datasets into a tuple format (?)
    """
    eval_data = []
    for name, eval_dataset in eval_datasets:
        if isinstance(eval_dataset, TspDataset):
            eval_data.append((name, torch.stack(eval_dataset[:], dim=0)))
        elif isinstance(eval_dataset, torch.Tensor):
            eval_data.append((name, eval_dataset))
        elif eval_dataset is not None:
            raise ValueError(f"Unrecognized eval_dataset type '{type(eval_dataset)}'")
    return eval_data


def init_distributed_torch(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def main(
    train_ctx,
    model_ctx,
    algo_ctx,
    train_dataset_ctx,
    eval_dataset_ctx,
    rank,
    world_size,
    gpu,
):

    if gpu is None:
        device_id = rank
    else:
        device_id = gpu

    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    init_distributed_torch(rank, world_size)

    eval_batch_size = (
        train_ctx.eval_batch_size
        if train_ctx.eval_batch_size is not None
        else train_ctx.batch_size
    )

    total_samples = ceil(1e3 * train_ctx.itr * train_ctx.batch_size)

    last_itr, last_time = setup_logging(
        rank, train_ctx.name, train_ctx.log_dir, train_ctx.resume
    )

    train_dataset = train_dataset_ctx(total_samples)
    dataloader = DataLoader(
        train_dataset, batch_size=train_ctx.batch_size, shuffle=True, num_workers=4
    )

    eval_data = eval_dataset_ctx(train_ctx.eval_samples)

    model = model_ctx(device)
    model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=True,
    )
    agent = TspAgent(model, device)
    algo = algo_ctx(rank, agent, eval_data[0][1])

    iteration = last_itr + 1

    eval_fn = partial(
        offline_eval_routine,
        eval_data,
        train_ctx.eval_period,
        train_ctx.eval_batch_size,
    )
    start_time = time()
    time_offset = 0
    log_fn = partial(
        log_step,
        rank,
        start_time,
        time_offset,
        train_ctx.keep_all_check,
        train_ctx.check_period,
    )

    dist.barrier()

    train(
        iteration,
        dataloader,
        agent,
        algo,
        eval_fn,
        log_fn,
    )


@dataclass(frozen=True)
class TrainingContext:
    name: str
    batch_size: int = 256
    eval_samples: int = 10000
    eval_batch_size: int = 1000
    itr: float = 250.0
    check_period: int = 100
    eval_period: int = 1000
    log_dir: str = None
    keep_all_check: bool = False
    resume: bool = False

    def __call__(self, model_ctx, algo_ctx, train_dataset_ctx, eval_dataset_ctx):
        return partial(
            main, self, model_ctx, algo_ctx, train_dataset_ctx, eval_dataset_ctx
        )

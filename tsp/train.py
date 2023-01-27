import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import os.path as osp
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tsp.logger import Logger, get_log_dir
from tsp.datagen import TspDataset
from tsp.eval import batched_eval
from tsp.agent import TspOracleAgent
from tsp.utils import RangeMap
from tsp.draw import plot_tsp


class TspTrainer:
    """
    TODO
    """

    def __init__(
        self, 
        dataset, 
        agent, 
        algo, 
        run_name=None, 
        eval_datasets=None,
        num_tours_draw=0, 
        log_dir=None, 
        resume=False
    ):
        """
        Provide TSP data, agent, and algorithm.

        The run_name param sepecifies the logging directory
        to be <install_path>/runs/<run_name>/.
        If this is left as None, no logging is performed.
        """
        self.dataset = dataset
        self.agent = agent
        self.algo = algo
        self.run_name = run_name

        # logging init
        if run_name is not None:

            if log_dir is None:
                log_dir = get_log_dir(run_name)

            output_path = osp.join(log_dir, run_name)
            self.last_itr, self.last_time = Logger.init(output_path, resume=resume)
        else:
            Logger.dummy_init()
            self.last_itr = self.last_time = 0

        self.eval_data = []
        # convert eval datasets to torch Tensors
        for name, eval_dataset in eval_datasets:
            if isinstance(eval_dataset, TspDataset):
                self.eval_data.append((name, torch.stack(eval_dataset[:], dim=0)))
            elif isinstance(eval_dataset, torch.Tensor):
                self.eval_data.append((name, eval_dataset))
            elif eval_dataset is not None:
                raise ValueError(
                    f"Unrecognized eval_dataset type '{type(eval_dataset)}'"
                )

        # check eval datasets contain enough samples for solution drawings
        for name, eval_dataset in self.eval_data:
            if len(eval_dataset) < num_tours_draw:
                raise ValueError(
                    f"Eval datset '{name}' has less samples than number of tours to be drawn: {len(eval_dataset)} / {num_tours_draw}"
                )
        self.num_tours_draw = num_tours_draw
    
    def _draw_setup(self):
        """
        Collect oracle solutions to drawn tours upfront
        and draw initial agent solution at initialization.
        """
        if not Logger.dummy:
            self.draw_save_path = osp.join(Logger.path_dir, "qual")
            if not osp.exists(self.draw_save_path):
                os.makedirs(self.draw_save_path)

            self.oracle_tours = {}
            oracle = TspOracleAgent()
            self.agent.eval_mode()

            for name, problems in self.eval_data:
                solutions, sol_idxs = oracle.solve(problems[:self.num_tours_draw])
                self.oracle_tours[name] = (solutions, sol_idxs)

                self._solve_draw_save_oracle_only(name, problems[:self.num_tours_draw], sol_idxs)
                self._solve_draw_save(0, name, problems[:self.num_tours_draw], sol_idxs)

    def _solve_draw_save(self, iteration, name, problems, oracle_tour_idxs):
        """
        Draw example solution tours with current agent state
        and save images to logging directory.
        """
        if not Logger.dummy:
            _, sel_idxs = self.agent.solve(problems)
            for idx in range(self.num_tours_draw):
                fig, ax = plt.subplots(figsize=(10, 10))
                plot_tsp(problems[idx], sel_idxs[idx], ax, oracle_tour_idxs[idx])

                save_path = osp.join(self.draw_save_path, f"{name}_{idx}_{iteration}.png")
                fig.savefig(save_path)
                plt.close(fig)

    def _solve_draw_save_oracle_only(self, name, problems, oracle_tour_idxs):
        if not Logger.dummy:
            for idx in range(self.num_tours_draw):
                fig, ax = plt.subplots(figsize=(10, 10))
                plot_tsp(problems[idx], oracle_tour_idxs[idx], ax, base_col="green")

                save_path = osp.join(self.draw_save_path, f"{name}_{idx}_oracle.png")
                fig.savefig(save_path)
                plt.close(fig)

    def _offline_eval_routine(self, iteration, eval_period, eval_batch_size, draw_period_map):
        """
        Run offline evaluation using all provided eval datasets.
        Also save tour drawings for a subset of each eval dataset,
        if specified.
        """
        for name, problems in self.eval_data:
            # quant eval
            if eval_period is not None and iteration % eval_period == 0:
                eval_costs = batched_eval(
                    self.agent, problems, batch_size=eval_batch_size
                )
                Logger.log_stat(f"eval_cost_{name}", eval_costs)
            elif eval_period is not None:
                Logger.log_stat(
                    f"eval_cost_{name}", None
                )  # null value to satisfy logger
            
            # qual eval
            if draw_period_map is not None and iteration % draw_period_map[iteration] == 0:
                _, sol_idxs = self.oracle_tours[name]
                self._solve_draw_save(iteration, name, problems[:self.num_tours_draw], sol_idxs)

    def start(
        self,
        epochs,
        batch_size,
        num_workers=0,
        eval_period=None,
        eval_batch_size=None,
        check_period=1,
        keep_all_check=False,
        draw_period_map=None,
    ):
        """
        Start training agent using algo and TSP dataset.

        epochs: number of times to cycle over the training datset
        batch_size: number of TSP tours to train with for one algo optimization step
        num_workers: number of subprocesses to spawn for dataloading during training (0 for main process only)
        eval_period: period between offline agent evaluation (None for no offline eval)
        eval_batch_size: number of TSP tours to use during offline agent evaluation (None defaults to training batch_size)
        check_period: checkpoint period in units of algo optimization steps
        keep_all_check: True if all model checkpoints are to be saved, False for only saving the latest
        draw_period_map: periods between drawing solution tours expressed as a utils.RangeMap object
        """
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        print(
            f"Training for {epochs} epochs of {len(dataloader)} algorithm iterations with {batch_size} batch-size"
        )

        if eval_period is not None:
            assert (
                self.eval_data is not None
            ), "Eval period set but no eval dataset provided during init"
            eval_batch_size = (
                eval_batch_size if eval_batch_size is not None else batch_size
            )
            print(
                f"Evaluating offline every {eval_period} iterations with {eval_batch_size} batch-size"
            )

        if draw_period_map is not None:
            assert isinstance(draw_period_map, RangeMap)
            assert self.num_tours_draw > 0, "Must specify number or tours to draw"
            print(f"Drawing solution tours at periods conditioned on iteration range: {draw_period_map}")
            print("Generating oracle solutions now...")
            self._draw_setup()
            print("...done")

        iteration = self.last_itr + 1
        time_offset = self.last_time
        start_time = time()
        for epoch_idx in range(epochs):

            for batch_idx, problem_batch in tqdm(enumerate(dataloader)):
                # train routine
                self.agent.train_mode()
                self.algo.optimize_agent(iteration, self.agent, problem_batch)

                # (offline) eval routine
                self.agent.eval_mode()
                self._offline_eval_routine(iteration, eval_period, eval_batch_size, draw_period_map)

                # log writing, checkpoint saving, etc.
                Logger.log("iteration", iteration)
                Logger.log("time", time() - start_time + time_offset)
                Logger.dump()

                if iteration % check_period == 0:
                    Logger.checkpoint(
                        self.agent.state_dict(), iteration, keep_prev=keep_all_check
                    )

                iteration += 1

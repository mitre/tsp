import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tsp.logger import Logger, get_log_dir
from tsp.datagen import TspDataset
from tsp.eval import batched_eval


class TspTrainer:
    """
    TODO
    """

    def __init__(self, dataset, agent, algo, run_name=None, eval_dataset=None):
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

        if run_name is not None:
            log_dir = get_log_dir(run_name)
            Logger.init(log_dir)
        else:
            Logger.dummy_init()

        if isinstance(eval_dataset, TspDataset):
            self.eval_problems = torch.stack(eval_dataset[:], dim=0)
        elif isinstance(eval_dataset, torch.Tensor):
            self.eval_problems = eval_dataset
        elif eval_dataset is not None:
            raise ValueError(f"Unrecognized eval_dataset type '{type(eval_dataset)}'")

    def start(
        self,
        epochs,
        batch_size,
        num_workers=0,
        eval_period=None,
        eval_batch_size=None,
        check_period=1,
        keep_all_check=False,
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
        """
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        print(
            f"Training for {epochs} epochs of {len(dataloader)} algorithm iterations with {batch_size} batch-size"
        )

        if eval_period is not None:
            assert (
                self.eval_problems is not None
            ), "Eval period set but no eval dataset provided during init"
            eval_batch_size = (
                eval_batch_size if eval_batch_size is not None else batch_size
            )
            print(
                f"Evaluating offline every {eval_period} iterations with {eval_batch_size} batch-size"
            )

        iteration = 1
        for epoch_idx in range(epochs):

            for batch_idx, problem_batch in tqdm(enumerate(dataloader)):
                # train routine
                self.agent.train_mode()
                self.algo.optimize_agent(self.agent, problem_batch)

                # (offline) eval routine
                self.agent.eval_mode()
                if eval_period is not None and iteration % eval_period == 0:
                    eval_costs = batched_eval(
                        self.agent, self.eval_problems, batch_size=eval_batch_size
                    )
                    Logger.log_stat("eval_cost", eval_costs)
                elif eval_period is not None:
                    Logger.log_stat("eval_cost", None)  # null value to satisfy logger

                # log writing, checkpoint saving, etc.
                Logger.log("iteration", iteration)
                Logger.dump()

                if iteration % check_period == 0:
                    Logger.checkpoint(
                        self.agent.state_dict(), iteration, keep_prev=keep_all_check
                    )

                iteration += 1

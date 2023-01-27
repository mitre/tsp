import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from tsp.utils import get_costs, get_entropy, pad_safe_dist_gather
from tsp.eval import batched_eval
from tsp.logger import Logger


class TspAlgoBase:
    """
    Base API for TSP RL algorithms.
    Expects a torch optimizer to be provided.
    """

    def __init__(self, rank, optimizer, grad_norm_clip="inf"):
        self.rank = rank
        self.optimizer = optimizer
        self.grad_norm_clip = grad_norm_clip

    def optimize_agent(self, iteration, agent, problems):
        """
        TODO
        """
        agent_outputs = agent(problems)
        loss = self.loss(*agent_outputs)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        Logger.log("loss", loss.item())
        Logger.log("grad_norm", grad_norm.item())


class TspReinforce(TspAlgoBase):
    """
    REINFORCE for TSP.

    Undiscounted in-between decisions,
    so each problem "episode" is treated
    as a factorized bandit problem.
    """

    def loss(self, selections, select_idxs, log_probs):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        tour_log_probs = torch.sum(sel_log_probs, dim=1)

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))

        return torch.mean(
            tour_costs * tour_log_probs
        )  # non-negative since using costs, not rewards


class TspA2C(TspReinforce):
    """
    Advantage Actor-Critic for TSP.

    A value is provided for each next-node
    decision based on all previous selection
    (the state in this context).

    The critic is trained to predict the
    future outcome of the current policy
    after rolling out the remaining selections.

    This is a factorized bandit problem,
    so no dicounting occurs. The value head
    learns the average tour cost generated
    from a given partial set of selections.
    """

    def __init__(self, optimizer, grad_norm_clip="inf", critic_coeff=1.0):
        super().__init__(optimizer, grad_norm_clip=grad_norm_clip)
        self.critic_coeff = critic_coeff

    def loss(self, selections, select_idxs, log_probs, values):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        tour_log_probs = torch.sum(sel_log_probs, dim=1)  # only for logging here

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

        # decision-level advantages
        # (removing last index based on value predictions where all selections have been made)
        advantages = (tour_costs_repeated - values)[:, :-1]

        Logger.log_stat("value", values.detach())
        Logger.log_stat("advantage", advantages.detach())

        actor_loss = torch.mean(
            advantages.detach() * sel_log_probs
        )  # non-negative since using costs, not rewards
        critic_loss = F.mse_loss(values, tour_costs_repeated)

        Logger.log("actor_loss", actor_loss.item())
        Logger.log("critic_loss", critic_loss.item())

        return actor_loss + self.critic_coeff * critic_loss


class TspA2CGreedyRollout(TspAlgoBase):
    """
    Performs A2C with greedy rollout baselines.
    """

    def __init__(
        self,
        rank,
        optimizer,
        eval_dataset,
        grad_norm_clip="inf",
        baseline_update_period=1000,
        eval_batch_size=512,
        tolerance=1e-3,
    ):
        super().__init__(rank, optimizer, grad_norm_clip)
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.tolerance = tolerance
        self.baseline_update_period = baseline_update_period

        self.critic_cost = None

    def optimize_agent(self, iteration, agent, problems):

        # TODO: This is ugly
        if self.critic_cost is None and self.rank == 0:
            critic_agent = agent.model.module.critic_model(agent.device)
            costs = batched_eval(
                critic_agent,
                torch.stack(self.eval_dataset.data),
                batch_size=self.eval_batch_size,
            )
            self.critic_cost = torch.mean(costs)

        agent_outputs = agent(problems)
        loss = self.loss(*agent_outputs)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        Logger.log("loss", loss.item())
        Logger.log("grad_norm", grad_norm.item())

        if iteration % self.baseline_update_period == 0:
            self._check_baseline_update(agent)

    def loss(self, selections, select_idxs, log_probs, crt_sel):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        tour_log_probs = torch.sum(sel_log_probs, dim=1)

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))

        values = get_costs(crt_sel)
        advantages = tour_costs - values

        Logger.log_stat("value", values.detach())
        Logger.log_stat("advantage", advantages.detach())

        return torch.mean(advantages * tour_log_probs)

    def _check_baseline_update(self, agent):
        if self.rank == 0:
            eval_costs = batched_eval(
                agent,
                torch.stack(self.eval_dataset.data),
                batch_size=self.eval_batch_size,
            )
            eval_cost = torch.mean(eval_costs)

            if eval_cost < (self.critic_cost + self.tolerance):
                agent.model.module.sync_baseline()
                self.critic_cost = eval_cost

        dist.barrier()
        agent.model._sync_params_and_buffers(authoritative_rank=0)


class TspSupervisedBase(TspAlgoBase):
    """
    Base API for TSP SL algorithms.
    """

    def optimize_agent(self, iteration, agent, problems):
        """
        TODO
        """
        data, labels = problems
        agent_outputs = agent.use(data, labels)
        loss = self.loss(*agent_outputs)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        Logger.log("loss", loss.item())
        Logger.log("grad_norm", grad_norm.item())


class TspActorCriticSupervised(TspSupervisedBase):
    """
    Actor-Critic for supervised TSP.

    Critic is unchanged from RL and tries to predict
    the undiscounted return of the actor policy.
    Actor is now trained through a NLL loss on
    the ground truth labels for each decision.
    """

    def __init__(self, optimizer, grad_norm_clip="inf", critic_coeff=1.0):
        super().__init__(optimizer, grad_norm_clip=grad_norm_clip)
        self.critic_coeff = critic_coeff

    def loss(self, selections, labels, log_probs, values):
        tour_costs = get_costs(selections)

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("value", values.detach())

        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        actor_loss = F.nll_loss(flat_log_probs, labels.flatten())

        critic_loss = F.mse_loss(values, tour_costs_repeated)

        Logger.log("actor_loss", actor_loss.item())
        Logger.log("critic_loss", critic_loss.item())

        return actor_loss + self.critic_coeff * critic_loss


class TspSupervisedGreedyRollout(TspSupervisedBase):
    def optimize_agent(self, iteration, agent, problems):
        """
        TODO Slow for larger models, just need to sync baseline
        before writing to disk, not every optimization iteration
        """
        super().optimize_agent(iteration, agent, problems)
        agent.model.sync_baseline()

    def loss(self, selections, select_idxs, log_probs, crt_sel):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        tour_log_probs = torch.sum(sel_log_probs, dim=1)

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))

        values = get_costs(crt_sel)
        Logger.log_stat("value", values.detach())

        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        actor_loss = F.nll_loss(flat_log_probs, select_idxs.flatten())

        return actor_loss

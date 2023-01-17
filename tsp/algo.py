import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from tsp.utils import get_costs, get_entropy, batch_dist_gather
from tsp.logger import Logger


class TspAlgoBase:
    """
    Base API for TSP RL algorithms.
    Expects a torch optimizer to be provided.
    """

    def __init__(self, optimizer, grad_norm_clip="inf"):
        self.optimizer = optimizer
        self.grad_norm_clip = grad_norm_clip

    def optimize_agent(self, agent, problems):
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

        sel_log_probs = batch_dist_gather(log_probs, select_idxs)
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

        sel_log_probs = batch_dist_gather(log_probs, select_idxs)
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

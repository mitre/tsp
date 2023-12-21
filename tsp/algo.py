import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np

from tsp.utils import get_costs, get_entropy, pad_safe_dist_gather
from tsp.logger import Logger


def iterate_mb_idxs(data_length, minibatch_size, shuffle=False):
    """
    Yields minibatches of indexes, to use as a for-loop iterator, with
    option to shuffle.

    Taken directly from rlpyt: https://github.com/astooke/rlpyt/blob/f04f23db1eb7b5915d88401fca67869968a07a37/rlpyt/utils/misc.py#L6
    """
    if shuffle:
        indexes = np.arange(data_length)
        np.random.shuffle(indexes)
    for start_idx in range(0, data_length - minibatch_size + 1, minibatch_size):
        batch = slice(start_idx, start_idx + minibatch_size)
        if shuffle:
            batch = indexes[batch]
        yield batch



class TspAlgoBase:
    """
    Base API for TSP RL algorithms.
    Expects a torch optimizer to be provided.
    """

    def __init__(self, optimizer, grad_norm_clip="inf"):
        self.optimizer = optimizer
        self.grad_norm_clip = grad_norm_clip

    def optimize_agent(self, agent, problems):
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


class TspPPO(TspA2C):
    """
    Proximal Policy Optimization for TSP

    Compositional w.r.t. values and policy distributions
    for node-wise decisions like TspA2C

    Only distinctions are 1) the PPO loss
    and 2) support for epochs of minibatch updates
    """
    def __init__(
            self, 
            optimizer, 
            epochs=4,
            minibatches=4,
            ratio_clip=1.0,
            grad_norm_clip="inf", 
            critic_coeff=1.0
        ):
        super().__init__(optimizer, grad_norm_clip=grad_norm_clip, critic_coeff=critic_coeff)
        
        self.epochs = epochs
        self.minibatches = minibatches
        self.ratio_clip = ratio_clip

    def _save_on_buffer(self, key, values):
        """
        Temporarily store and aggregate logging signals throughout minibatches
        """
        if not hasattr(self, "_buff"):
            self._buff = dict()

        def deactivate(tensor):
            return tensor.flatten().detach().cpu()

        self._buff[key] = torch.cat((self._buff[key], deactivate(values))) if key in self._buff else deactivate(values)

    def _log_buffer(self):
        """
        Log all infos stored through _save_on_buffer()
        """
        for key, values in self._buff.items():
            Logger.log_stat(key, values)

        self._buff = dict()

    def optimize_agent(self, agent, problems):
        # gather pre-determined selection sequences for PPO minibatch updates (can't re-roll-out each time when comparing with old_log_probs in PPO objective)
        # and gather starting policy distributions for PPO importance sampling ratio
        with torch.no_grad():
            _, old_select_idxs, old_log_probs, old_values = agent(problems)

        # determine minibatch sample size
        batch_size = len(problems)
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):

            for mb_idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                # extract minibatch of inputs for agent and loss
                problems_mb = problems[mb_idxs]
                select_idxs_mb = old_select_idxs[mb_idxs]
                old_log_probs_mb = old_log_probs[mb_idxs]
                old_values_mb = old_values[mb_idxs]

                # compute model outputs with gradient tracking
                selections, re_select_idxs, log_probs, values = agent.use(problems_mb, select_idxs_mb)

                # double check model took the same subdecision rollouts as old_select_idxs (very important)
                assert torch.allclose(select_idxs_mb.float(), re_select_idxs.float())

                # compute PPO clip loss
                loss = self.loss(selections, select_idxs_mb, log_probs, old_log_probs_mb, values, old_values_mb)

                # optimization step
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
                self.optimizer.step()

                # logging
                self._save_on_buffer("loss", loss)
                self._save_on_buffer("grad_norm", grad_norm)

        self._log_buffer()

    def loss(self, selections, select_idxs, log_probs, old_log_probs, values, old_values):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        old_sel_log_probs = pad_safe_dist_gather(old_log_probs, select_idxs, reset_val=0.0)

        tour_log_probs = torch.sum(sel_log_probs, dim=1)  # only for logging here
        
        self._save_on_buffer("train_cost", tour_costs)
        self._save_on_buffer("log_pi", tour_log_probs)
        self._save_on_buffer("entropy", get_entropy(log_probs))
        self._save_on_buffer("pi_drift", torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs))
        self._save_on_buffer("abs_pi_drift", torch.abs(torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs)))

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

        # decision-level advantages
        # (removing last index based on value predictions where all selections have been made)
        advantages = (tour_costs_repeated - old_values)[:, :-1]

        self._save_on_buffer("value", values)
        self._save_on_buffer("advantage", advantages)
        self._save_on_buffer("value_drift", values.detach() - old_values)
        self._save_on_buffer("abs_value_drift", torch.abs(values.detach() - old_values))

        # compute PPO clip ratio and surrogate objective (old_log_probs should already have no tracked gradients, but detaching just in case)
        importance_ratio = torch.exp(sel_log_probs - old_sel_log_probs.detach())
        clipped_ratio = torch.clamp(importance_ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)

        surr_1 = importance_ratio * advantages.detach()  # advantages should already not have tracked gradients, but detaching just to be safe
        surr_2 = clipped_ratio * advantages.detach()
        surrogate_objective = -torch.max(surr_1, surr_2)  # -max(-x, -y) == min(x, y) --> surrogates actually have sign flipped since we use costs, not rewards

        # compute PPO clip loss over non-pad (i.e. gradient-tracked actor and critic) decisions
        actor_loss = -torch.mean(surrogate_objective)
        critic_loss = F.mse_loss(values, tour_costs_repeated)

        self._save_on_buffer("actor_loss", actor_loss)
        self._save_on_buffer("critic_loss", critic_loss)

        return actor_loss + self.critic_coeff * critic_loss


class TspSupervisedBase(TspAlgoBase):
    """
    Base API for TSP SL algorithms.
    """

    def optimize_agent(self, agent, problems):
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

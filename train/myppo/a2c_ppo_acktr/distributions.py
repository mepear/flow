import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from a2c_ppo_acktr.utils import AddBias, init
from .utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, **kwargs):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class FixedMultiCategorical:

    def __init__(self, logits=None, action_dims=None):
        self.action_dims = action_dims
        self.distributions = [torch.distributions.Categorical(logits=split) for split in torch.split(logits, tuple(self.action_dims), dim=1)]

    def log_probs(self, actions):
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1).unsqueeze(-1)

    def entropy(self):
        return torch.stack(
            [dist.entropy() for dist in self.distributions], dim=1
        ).sum(dim=1)

    def sample(self):
        return torch.stack(
            [dist.sample() for dist in self.distributions], dim=1
        )

    def mode(self):
        # print('=' * 10, [torch.max(dist.probs, dim=1)[0] for dist in self.distributions])
        return torch.stack(
            [torch.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1
        )


class MultiCategorical(nn.Module):
    
    def __init__(self, num_inputs, action_dims):
        super(MultiCategorical, self).__init__()
        self.action_dims = action_dims
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, sum(action_dims)))

    def forward(self, x, mask=None):
        x = self.linear(x)
        if mask is not None:
            assert x.size() == mask.size()
            x[mask] = -float('inf')
        return FixedMultiCategorical(logits=x, action_dims=self.action_dims)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.to(x.device)

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

from dataclasses import dataclass

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


def decoding_sampler(logits, mode, tau=1, hard=False, dim=-1):
    y_soft = None

    if mode == 'REINFORCE' or mode == 'SCST':
        cat_distr = OneHotCategorical(logits=logits)
        return cat_distr.sample()
    elif mode == 'GUMBEL':
        cat_distr = RelaxedOneHotCategorical(tau, logits=logits)
        y_soft = cat_distr.rsample()
    elif mode == 'SOFTMAX':
        y_soft = F.softmax(logits, dim=1)
    
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, device=args.device).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparameterisation trick.
        ret = y_soft

    return ret


@dataclass
class Configurations:
    hidden_size = 256
    dropout_ratio = 0.8
    tau = 1.0 # tau for GUMBLE trick
    msg_mode = 'GUMBEL' # options: GUMBEL, REINFORCE, SCST, SOFTMAX
    max_msg_len = 4 # maximum length of the messages

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


args = Configurations()
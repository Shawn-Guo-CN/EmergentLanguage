from agents.utils import Configurations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import optim


@dataclass
class Configurations(object):
    """Default configurations for classes in module 'games'.
    """
    optimiser = optim.Adam
    learning_rate = 1e-3
    tau = 1.0 # hyper-param for Gumbel trick
    clip = 50.0

args = Configurations()
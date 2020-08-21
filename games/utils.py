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

args = Configurations()
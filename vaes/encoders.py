import torch
import torch.nn as nn

from utils import View, kaiming_init


class BaseEncoder(nn.Module):
    """TODO: implement this base encoder in the future to load and save model with json file."""


class DspriteImgCNNEncoder(nn.Module):
    """The specific CNN encoder for dSprite images."""
    
    def __init__(self, z_dim=10) -> None:
        super().__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.weight_init()

    def forward(self, x):
        return self.encoder(x)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


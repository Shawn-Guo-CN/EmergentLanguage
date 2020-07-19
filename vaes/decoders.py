import torch
import torch.nn as nn

from utils import View, kaiming_init


class DspriteImgCNNDecoder(nn.Module):
    """The specific CNN encoder for dSprite images."""
    
    def __init__(self, z_dim=10) -> None:
        super().__init__()

        self.z_dim = z_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), # B,  nc, 64, 64
        )

        self.weight_init()

    def forward(self, x):
        return self.encoder(x)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
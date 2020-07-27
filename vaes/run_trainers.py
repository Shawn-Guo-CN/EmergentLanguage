from .args import get_dsprites_betavae_args
from .models import DspritesBetaVAETrainer


def train_dSprites_betavae():
    args = get_dsprites_betavae_args()
    trainer = DspritesBetaVAETrainer(args)
    trainer.train()

train_dSprites_betavae()

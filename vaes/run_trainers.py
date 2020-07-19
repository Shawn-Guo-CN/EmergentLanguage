from args import get_dsprites_betavae_args
from vaes import DspritesBetaVAETrainer


def train_dSprites_betavae():
    args = get_dsprites_betavae_args()
    trainer = DspritesBetaVAETrainer(args)
    trainer.train()


if __name__ == '__main__':
    train_dSprites_betavae()
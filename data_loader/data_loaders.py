import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .data_sets import DSpritesDataset
from .data_transforms import DSpritesToTensor


def get_dSprite_loader(npz_file, batch_size, num_workers, shuffle=True):
    transformed_dataset = DSpritesDataset(npz_file=npz_file, transform=transforms.Compose([DSpritesToTensor()]))
    return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

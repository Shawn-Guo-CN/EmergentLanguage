import numpy as np
import torch
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    """dSprites dataset."""

    def __init__(self, npz_file:str, transform=None):
        """
        Args:
            npz_file: Path to the npz file.
            root_dir: Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset_zip = np.load(npz_file, allow_pickle=True, encoding='latin1')
        self.dataset = self.preprocess_zip(dataset_zip)
        del dataset_zip

        self.transform = transform

    def __len__(self):
        return self.dataset['imgs'].shape[0]

    def __getitem__(self, idx):
        img = self.dataset['imgs'][idx]
        latents_class = self.dataset['latents_classes'][idx]
        latents_value = self.dataset['latents_values'][idx]
        return img, latents_class, latents_value

    @staticmethod
    def preprocess_zip(data_zip):
        # TODO: filter out the data we do not need in the future

        return {
            'imgs': data_zip['imgs'],
            'latents_classes': data_zip['latents_values'],
            'latents_values': data_zip['latents_values']
        }


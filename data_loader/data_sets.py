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
        return self.dataset['images'].shape[0]

    def __getitem__(self, idx):
        image = self.dataset['images'][idx]
        latents_class = self.dataset['latents_classes'][idx]
        latents_value = self.dataset['latents_values'][idx]

        sample = (image, latents_class, latents_value)

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def preprocess_zip(data_zip):
        # TODO: filter out the data we do not need in the future

        return {
            'images': data_zip['imgs'],
            'latents_classes': data_zip['latents_values'],
            'latents_values': data_zip['latents_values']
        }


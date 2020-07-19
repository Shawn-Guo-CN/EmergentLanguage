import torch


class DSpritesToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        (image, latents_class, latents_value) = sample

        return {
            'images': torch.from_numpy(image).to(torch.float).view(1, 64, 64),
            'latents_classes': torch.from_numpy(latents_class),
            'latents_values': torch.from_numpy(latents_value)
        }
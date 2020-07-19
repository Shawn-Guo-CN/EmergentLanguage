from data_loaders import get_dSprite_loader


def test_dSprite_loader():
    loader = get_dSprite_loader(
        npz_file='data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
        batch_size=32,
        num_workers=0
    )

    for idx, data in enumerate(loader):
        image = data['images']
        latent_class = data['latents_classes']
        latent_value = data['latents_values']
        print(image)
        print(latent_class)
        print(latent_value)
        
        if idx >= 1:
            break


if __name__ == '__main__':
    test_dSprite_loader()
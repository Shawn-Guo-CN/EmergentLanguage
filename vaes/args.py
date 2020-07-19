import argparse

from utils import str2bool


def get_dsprites_betavae_args():
    """codes from https://github.com/1Konny/Beta-VAE/blob/master/utils.py"""

    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--npz_file', default='data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', type=str, help='dSprites npz file path')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')

    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--display_step', default=10, type=int, help='number of iterations after which loss data is printed')

    parser.add_argument('--ckpt_dir', default='data/checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='dsprites_betavae', type=str, help='load previous checkpoint. insert checkpoint filename')

    args = parser.parse_args()

    return args
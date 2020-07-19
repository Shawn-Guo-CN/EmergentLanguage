import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from encoders import DspriteImgCNNEncoder
from decoders import DspriteImgCNNDecoder
from utils import reparameterise, reconstruction_loss, kl_divergence

# TODO: fix the absolute import below
import sys
sys.path.insert(0,'E:\\GitWS\\EmergentLanguage\\')
from data_loader import get_dSprite_loader



class DspritesVAE(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = DspriteImgCNNEncoder(self.z_dim)
        self.decoder = DspriteImgCNNDecoder(self.z_dim)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparameterise(mu, logvar)
        x_recon = self.decoder(z).view(x.size())

        return x_recon, mu, logvar


class DspritesBetaVAETrainer(object):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.decoder_dist = 'bernoulli'

        self.model = DspritesVAE(self.z_dim).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        self.ckpt_dir = os.path.join(args.ckpt_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_step = args.save_step
        self.display_step = args.display_step

        self.npz_file = args.npz_file
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_loader = get_dSprite_loader(self.npz_file, self.batch_size, self.num_workers)

    def train(self):
        self.model.train()
        self.C_max = Variable(torch.FloatTensor([self.C_max]).to(self.device))
        finish = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not finish:
            for sample in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                img = sample['images'].to(self.device)
                img_recon, mu, logvar = self.model(img)
                recon_loss = reconstruction_loss(img, img_recon, self.decoder_dist)
                total_kld, _, mean_kld = kl_divergence(mu, logvar)

                C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.global_iter % self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)
                    pbar.write('C:{:.3f}'.format(C.data[0]))

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter >= self.max_iter:
                    finish = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'model':self.model.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {
            'iter':self.global_iter,
            'model_states':model_states,
            'optim_states':optim_states
            }

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.model.load_state_dict(checkpoint['model_states']['model'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

from abc import ABC, abstractstaticmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import args
from .encoders import SeqEncoderLSTM, ImgEncoderCNN


class BaseReferListener(nn.Module, ABC):
    def __init__(self, msg_vocsize, hidden_size):
        super().__init__()
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size

        self.msg_embedding = nn.Embedding(self.msg_vocsize, self.hidden_size).weight

        self.msg_encoder, self.can_encoder = self._init_encoders()

    def forward(self, message, msg_mask, candidates):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        message = F.relu(torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1)))

        _, msg_encoder_hidden, _ = self.msg_encoder(message, msg_len)
        msg_encoder_hidden = msg_encoder_hidden.transpose(0, 1).transpose(1, 2)

        can_encoder_hiddens = []
        for candidate in candidates:
            input_imgs = candidate['imgs']
            encoder_hidden = self.can_encoder(input_imgs)
            can_encoder_hiddens.append(encoder_hidden)

        can_encoder_hiddens = torch.stack(can_encoder_hiddens).transpose(0, 1)

        choose_logits = torch.bmm(can_encoder_hiddens, msg_encoder_hidden).squeeze(2)
        return choose_logits

    @abstractstaticmethod
    def _init_encoders():
        raise NotImplementedError


class BaseReconListener(nn.Module, ABC):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.msg_embedding = nn.Embedding(self.input_size, self.hidden_size).weight

        self.encoder, self.decoder = self._init_submodules()

    def forward(self, message, msg_mask, target_max_len):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        message = F.relu(torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1)))

        _, encoder_hidden, encoder_cell = self.encoder(message, msg_len)
        encoder_hidden = encoder_hidden.squeeze(0)
        encoder_cell = encoder_cell.squeeze(0)

        recons = self.decoder(encoder_hidden, encoder_cell)

        return recons

    @abstractstaticmethod
    def _init_submodules():
        raise NotImplementedError

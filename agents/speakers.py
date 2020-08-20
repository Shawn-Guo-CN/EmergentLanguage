import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import args
from .encoders import ImgEncoderCNN, SeqEncoderLSTM
from .decoders import SeqDecoderLSTM


class ImgCnnSpeakerLSTM(nn.Module):
    def __init__(
            self, msg_vocsize,
            hidden_size=args.hidden_size,
            dropout=args.dropout_ratio,
            msg_length=args.max_msg_len,
            msg_mode=args.msg_mode
        ):
        super().__init__()
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.msg_length = msg_length
        self.msg_mode = msg_mode

        self.msg_embedding = nn.Embedding(self.msg_vocsize, self.hidden_size).weight

        self.encoder = ImgEncoderCNN(self.hidden_size)
        # The output size of decoder is the size of vocabulary for communication
        self.decoder = SeqDecoderLSTM(
            self.msg_vocsize, self.hidden_size, self.msg_vocsize,
            embedding=self.msg_embedding, role='msg'
        )

    def forward(self, imgs, tau=1.0):
        img_hidden = self.encoder(imgs)
        message, logits, mask = self.decoder(
                img_hidden, img_hidden, self.msg_length,
                mode=self.msg_mode, sample_tau=tau
            )

        return message, logits, mask

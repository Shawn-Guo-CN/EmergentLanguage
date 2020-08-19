import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import args


class SeqEncoderLSTM(nn.Module):
    def __init__(self, in_size=args.hidden_size, hidden_size=args.hidden_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(in_size, hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_embedded, input_lengths):
        """
        inpug_embedded shape: [B, L, I]
        input_lengths shape: [B]
        """
        batch_size = input_embedded.shape[0]

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            input_embedded, input_lengths, 
            batch_first=True, enforce_sorted=False)

        # Forward pass through LSTM
        h0 = self.init_hidden.expand(1, batch_size, -1).contiguous()
        c0 = self.init_cell.expand(1, batch_size, -1).contiguous()
        outputs, (hidden, cell) = self.lstm(packed, (h0, c0))

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Return output and final hidden state
        return outputs, hidden, cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=args.device))


class ImgEncoderCNN(nn.Module):
    def __init__(self, hidden_size=args.hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # input has to be 3 channels
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # the following size is only feasible for images with size 3*100*50
        self.fc1 = nn.Linear(16 * 22 * 9, 1024)
        self.fc2 = nn.Linear(1024, self.hidden_size)

    def forward(self, imgs_tensor):
        x = F.max_pool2d(F.relu(self.conv1(imgs_tensor)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 22 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

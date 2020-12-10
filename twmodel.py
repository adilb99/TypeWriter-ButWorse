import torch
import torch.nn as nn
import numpy as np


class TWModel(nn.Module):
    """
    This is the complete neural model of TypeWriter
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(TWModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_occ = nn.LSTM(5, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.lstm_fb = nn.LSTM(113, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)
        self.lstm_doc = nn.LSTM(100, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)

        self.linear = nn.Linear(hidden_size * 4 * 2, self.output_size)

    def forward(self, x_fb, x_doc, x_occ):
        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_occ.flatten_parameters()
        self.lstm_fb.flatten_parameters()
        self.lstm_doc.flatten_parameters()

        x_occ, _ = self.lstm_occ(x_occ)
        x_doc, _ = self.lstm_doc(x_doc)
        x_fb, _ = self.lstm_fb(x_fb)

        # Decode the hidden state of the last time step
        x_occ = x_occ[:, -1, :]
        x_doc = x_doc[:, -1, :]
        x_fb = x_fb[:, -1, :]

        x = torch.cat((x_fb, x_doc, x_occ), 1)

        x = self.linear(x)
        return x

    def reset_model_parameters(self):
        """
        This resets all the parameters of the model.
        It would be useful to train the model for several trials.
        """
        self.lstm_doc.reset_parameters()
        self.lstm_occ.reset_parameters()
        self.lstm_fb.reset_parameters()
        self.linear.reset_parameters()


def test():
    input_size = 100
    hidden_size = 100
    output_size = 1000
    num_layers = 1
    
    model = TWModel(input_size, hidden_size, num_layers, output_size)
    print(model)
    '''
    Returns:
    body: 127x40x113
    docstring: 100x100
    occurences: 5x1
    label: 2000x1 
    
    Arguments:
    body: 130x40x113
    docstring: 100x100
    occurences: 5x1
    label: 2000x1
    '''
    x_occ = torch.randn(5, 1, 5)
    x_fb = torch.randn(5, 130*40, 113)

    x_doc= torch.randn(5, 100, 100)

    x_out = model(x_occ, x_doc, x_fb)
    print(x_out.shape)

test()

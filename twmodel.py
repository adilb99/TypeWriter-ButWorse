import torch
import torch.nn as nn
import numpy as np


class TWModel(nn.Module):
    """
    This is the complete neural model of TypeWriter
    """

    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int, num_layers: int, output_size: int):
        super(TWModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)
        self.lstm_cm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)

        self.linear = nn.Linear(hidden_size * 3 * 2 + self.aval_type_size, self.output_size)

    def forward(self, x_id, x_tok, x_cm, x_type):

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()
        self.lstm_cm.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)
        x_cm, _ = self.lstm_cm(x_cm)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]
        x_cm = x_cm[:, -1, :]

        x = torch.cat((x_id, x_cm, x_tok, x_type), 1)

        x = self.linear(x)
        return x

    def reset_model_parameters(self):
        """
        This resets all the parameters of the model.
        It would be useful to train the model for several trials.
        """

        self.lstm_id.reset_parameters()
        self.lstm_cm.reset_parameters()
        self.lstm_tok.reset_parameters()
        self.linear.reset_parameters()


def test():
    input_size = 100
    hidden_size = 100
    output_size = 1000
    num_layers = 1
    
    model = TWModel(input_size, hidden_size, input_size, num_layers, output_size)
    print(model)
    '''
        x_type : (num_types, input_size)
        x_type : (num_types, , input_size)
        x_type : (num_types, , input_size)
        x_type : (num_types, , input_size)
    '''
    x_id = torch.randn(5, 3, input_size)
    x_tok = torch.randn(5, 3, input_size)
    x_cm = torch.randn(5, 3, input_size)
    x_type = torch.randn(5, input_size)

    x_out = model(x_id, x_tok, x_cm, x_type)
    print(x_out.shape)

test()

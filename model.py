import torch.nn as nn


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn_class):
        super(ANNModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.hidden_layer = nn.Linear(input_dim, hidden_dim+1, bias=False)
        self.output_layer = nn.Linear(hidden_dim+1, output_dim, bias=False)
        self.activation_fn = activation_fn_class()

    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.output_layer(out)
        return out

    def get_weights(self):
        return (
            self.state_dict()["hidden_layer.weight"].numpy().T,
            self.state_dict()["output_layer.weight"].numpy().T,
        )

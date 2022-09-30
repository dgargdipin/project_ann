import torch.nn as nn


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn_class):
        super(ANNModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.hidden_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
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

    def train_with_labels(self, iterations, optimizer, inp_data, labels, error):
        loss_list = []
        for iteration in range(iterations):
            # optimization
            optimizer.zero_grad()

            # Forward to get output
            results = self(inp_data)

            # Calculate Loss
            loss = error(results, labels)

            # backward propagation
            loss.backward()

            # Updating parameters
            optimizer.step()

            # store loss
            loss_list.append(loss.data.cpu())

            # print loss
            if iteration % 100 == 0:
                print("epoch {}, loss {}".format(iteration, loss.data))
        return loss_list

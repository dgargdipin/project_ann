from generate_data import convert_np_to_torch, generate_data_np
from model import ANNModel
import torch

from weights import connection_weight, garsons, perturbation


def monte_carlo_simulation(
    num_simulations,
    device,
    input_dim,
    hidden_dim,
    output_dim,
    activation_fn,
    learning_rate=0.02,
):
    connection_importance_list = []
    for simulation in range(num_simulations):
        inp_data, labels = convert_np_to_torch(device, *generate_data_np())

        net = ANNModel(input_dim, hidden_dim, output_dim, activation_fn)
        net = net.to(device)

        error = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        net.train_with_labels(
            iterations=1000,
            optimizer=optimizer,
            inp_data=inp_data,
            labels=labels,
            error=error,
        )
        connection_importance_curr = {}
        connection_importance_curr["connection_weight"] = connection_weight(net)
        connection_importance_curr["garsons"] = garsons(net)
        connection_importance_curr["perturbation"] = perturbation(net)
        connection_importance_list.append(connection_importance_curr)
    return connection_importance_list

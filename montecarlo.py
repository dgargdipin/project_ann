from generate_data import convert_np_to_torch, generate_data_np
from model import ANNModel
import torch
import numpy as np
from weights import connection_weight, garsons, perturbation, rank_weights


def monte_carlo_simulation(
    num_simulations,
    device,
    input_dim,
    hidden_dim,
    output_dim,
    activation_fn,
    learning_rate=0.02,
    num_sample_rows=50,
):
    connection_importance_list = []
    for simulation in range(num_simulations):
        inp_data_np, labels_np = generate_data_np()
        random_rows = np.random.choice(
            inp_data_np.shape[0], num_sample_rows, replace=False
        )
        inp_data_np_filtered = inp_data_np[random_rows]
        labels_np_filtered = labels_np[random_rows]
        input_data, labels = convert_np_to_torch(
            device, inp_data=inp_data_np_filtered, labels=labels_np_filtered
        )

        net = ANNModel(input_dim, hidden_dim, output_dim, activation_fn)
        net = net.to(device)

        error = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        net.train_with_labels(
            iterations=1000,
            optimizer=optimizer,
            inp_data=input_data,
            labels=labels,
            error=error,
        )
        connection_importance_curr = {}
        connection_importance_curr["connection_weight"] = rank_weights(
            connection_weight(net)
        )
        connection_importance_curr["garsons"] = rank_weights(garsons(net))
        connection_importance_curr["perturbation"] = rank_weights(
            perturbation(net, input_data, labels, error)
        )
        connection_importance_list.append(connection_importance_curr)
    return connection_importance_list

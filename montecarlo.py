from generate_data import convert_np_to_torch, generate_data_np
from model import ANNModel
import torch
import numpy as np
from weights import (
    connection_weight,
    garsons,
    perturbation,
    rank_weights,
    similarity_coefficient,
)


def monte_carlo_simulation(
    num_simulations,
    device,
    input_dim,
    hidden_dim,
    output_dim,
    activation_fn,
    actual_importance,
    learning_rate=0.02,
    num_sample_rows=50,
):
    connection_importance = {}
    connection_importance["connection_weight"] = 0
    connection_importance["garsons"] = 0
    connection_importance["perturbation"] = 0
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
            iterations=500,
            optimizer=optimizer,
            inp_data=input_data,
            labels=labels,
            error=error,
        )
        connection_importance["connection_weight"] += similarity_coefficient(
            rank_weights(connection_weight(net)), actual_importance
        )
        connection_importance["garsons"] += similarity_coefficient(
            rank_weights(garsons(net)), actual_importance
        )
        connection_importance["perturbation"] += similarity_coefficient(
            rank_weights(perturbation(net, input_data, labels, error)),
            actual_importance,
        )
    connection_importance["connection_weight"] /= num_simulations
    connection_importance["garsons"] /= num_simulations
    connection_importance["perturbation"] /= num_simulations
    return connection_importance

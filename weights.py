import numpy as np
import torch


def connection_weight(model):
    hidden_layer_weights, output_layer_weights = model.get_weights()
    input_dim, hidden_dim = hidden_layer_weights.shape
    connection_weights = []
    for i in range(input_dim):
        curr_sum = 0
        for j in range(hidden_dim):
            curr_sum += hidden_layer_weights[i][j] * output_layer_weights[j][0]
        connection_weights.append(curr_sum)
    return connection_weights


def garsons(model):
    hidden_layer_weights, output_layer_weights = model.get_weights()
    input_dim, hidden_dim = hidden_layer_weights.shape
    hl_mod_weights = np.abs(hidden_layer_weights)
    col_sums = np.sum(hl_mod_weights, axis=0).reshape(1, -1)
    Q_arr = hl_mod_weights / col_sums
    Q_sum = np.sum(Q_arr)
    return np.sum(Q_arr, axis=1) * 100 / Q_sum


def perturbation(model, input_data, labels, error):
    model.eval()
    input_dim = model.hidden_layer.in_features
    ans = []
    with torch.no_grad():
        for i in range(input_dim):
            noise = np.random.normal(0, 1, 10000) * 0.5
            inp_copy = input_data.numpy().copy().T
            inp_copy[i] += noise
            altered_inp = torch.from_numpy(inp_copy.T)
            results_init = model(input_data)
            loss_init = error(results_init, labels)
            results_final = model(altered_inp)
            loss_final = error(results_final, labels)
            ans.append((loss_final - loss_init).numpy())
    return ans

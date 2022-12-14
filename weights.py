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
    return np.array(connection_weights)


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
            noise = np.random.normal(0, 1, input_data.shape[0]) * 0.5
            inp_copy = input_data.numpy().copy().T
            inp_copy[i] += noise
            altered_inp = torch.from_numpy(inp_copy.T)
            results_init = model(input_data)
            loss_init = error(results_init, labels)
            results_final = model(altered_inp)
            loss_final = error(results_final, labels)
            ans.append((loss_final - loss_init).numpy())
    return np.array(ans)


def rank_weights(weights):
    return weights.argsort()[::-1] + 1


def similarity_coefficient(arr1, arr2):
    range_arr = np.max(arr1) - np.min(arr1)
    assert len(arr1) == len(arr2)
    index_arr1 = [0] * len(arr1)
    index_arr2 = [0] * len(arr2)
    for i in range(len(arr1)):
        index_arr1[arr1[i] - 1] = i
        index_arr2[arr2[i] - 1] = i
    ans = 0
    for i in range(len(arr1)):
        ans += 1 - abs(index_arr1[i] - index_arr2[i]) / range_arr
    return ans / len(arr1)

import os
import torch.nn as nn
import torch
from model import ANNModel
from generate_data import generate_data
from weights import connection_weight, garsons, perturbation
from plots import save_losses

INPUT_DIM = 5
HIDDEN_DIM = 5
OUTPUT_DIM = 1
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

inp_data, labels = generate_data(DEVICE)

net = ANNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, nn.ReLU)
net = net.to(DEVICE)

error = torch.nn.MSELoss()
learning_rate = 0.02
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


loss_list = []
iteration_number = 1000
for iteration in range(iteration_number):
    # optimization
    optimizer.zero_grad()

    # Forward to get output
    results = net(inp_data)

    # Calculate Loss
    loss = error(results, labels)

    # backward propagation
    loss.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    loss_list.append(loss.data.cpu())

    # print loss
    if iteration % 50 == 0:
        print("epoch {}, loss {}".format(iteration, loss.data))


print(perturbation(net, input_data=inp_data, labels=labels, error=error))

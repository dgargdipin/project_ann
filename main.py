import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model import ANNModel
from generate_data import generate_data
import time

INPUT_DIM = 5
HIDDEN_DIM = 5
OUTPUT_DIM = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

inp_data_np, labels_np = generate_data()

inp_data = torch.from_numpy(inp_data_np).float().cuda()

labels = torch.from_numpy(labels_np).float().unsqueeze(1).cuda()


net = ANNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, nn.ReLU)
net = net.to(DEVICE)

error = torch.nn.MSELoss()
learning_rate = 0.02
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


loss_list = []
iteration_number = 1001
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


if not os.path.exists('plots'):
    os.makedirs('plots')

import matplotlib
matplotlib.use('Agg')
plt.plot(range(iteration_number), loss_list)
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
plt.savefig(os.path.join('plots',timestr+'.png'))

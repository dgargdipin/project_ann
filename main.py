import os
import torch.nn as nn
import torch
from model import ANNModel
from generate_data import (
    convert_np_to_torch,
    generate_data_np,
)
from montecarlo import monte_carlo_simulation
from weights import connection_weight, garsons, perturbation
from plots import save_losses

INPUT_DIM = 5
HIDDEN_DIM = 5
OUTPUT_DIM = 1
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


def train_model():
    inp_data, labels = convert_np_to_torch(DEVICE, *generate_data_np())

    net = ANNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, nn.ReLU)
    net = net.to(DEVICE)

    error = torch.nn.MSELoss()
    learning_rate = 0.02
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    net.train_with_labels(
        iterations=1000,
        optimizer=optimizer,
        inp_data=inp_data,
        labels=labels,
        error=error,
    )

monte_carlo_simulation(500,DEVICE,INPUT_DIM,HIDDEN_DIM,OUTPUT_DIM,nn.ReLU)
import os
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,output_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

capacity = 32
cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")

class Encoder(nn.Module):
    
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.fc_mean = torch.nn.Linear(c * 2 * 7 * 7, latent_dim)
        self.fc_var = torch.nn.Linear(c * 2 * 7 * 7, latent_dim)
        
        self.training = True
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        c = capacity
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        if x.dim() == 1:
            x = x.view(1, capacity * 2, 7, 7)
        else:
            x = x.view(x.size(0), capacity * 2, 7, 7)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.conv1(x))
        return x
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
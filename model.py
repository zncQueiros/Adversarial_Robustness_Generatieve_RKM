import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# Hyper-parameters =================
capacity = 32
x_fdim = 128
y_fdim = 20


# Feature-map/encoder - network architecture
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(c * 2 * 7 * 7, x_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = torch.nn.Linear(10, 15)
        self.fc2 = torch.nn.Linear(15, y_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)
        return x


# Pre-image map/decoder - network architecture
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        c = capacity
        self.fc1 = nn.Linear(in_features=x_fdim, out_features=c * 2 * 7 * 7)
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


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = torch.nn.Linear(y_fdim, 15)
        self.fc2 = torch.nn.Linear(15, 10)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = torch.sigmoid(self.fc2(x))
        return x


# noise_scale = 0.05


# class Model(nn.Module):
#     def __init__(self, Encoder, Decoder):
#         super(Model, self).__init__()
#         self.Encoder = Encoder
#         self.Decoder = Decoder
#
#     def reparameterization(self, mean, var):
#         epsilon = torch.randn_like(var).to(device)  # sampling epsilon
#         z = mean + var * epsilon  # reparameterization trick
#         return z
#
#     def forward(self, x):
#         mean, log_var = self.Encoder(x)
#         z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
#         z_noise = z + noise_scale * torch.randn_like(z)  # adding noise on latent space
#         x_hat = self.Decoder(z_noise)
#
#         return x_hat, mean, log_var
#
#
# model = Model(Encoder=net1, Decoder=net3).to(device)

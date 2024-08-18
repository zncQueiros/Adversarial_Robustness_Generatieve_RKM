import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from utils import *
import numpy as np
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', type=str, default='pre_trained_models/MNIST_trained_RKM_f_bce', help='Enter Filename')
parser.add_argument('--device', type=str, default='cpu', help='Device type: cuda or cpu')
opt_gen = parser.parse_args()

sd_mdl = torch.load('{}.tar'.format(opt_gen.filename), map_location=torch.device('cpu'))

for key, value in sd_mdl.items():
    print(key)

print(sd_mdl['capacity'])
print(sd_mdl['N'])
print(sd_mdl['s'].shape)
print(sd_mdl['U'].shape)
print(sd_mdl['V'].shape)
print(sd_mdl['h'].shape)
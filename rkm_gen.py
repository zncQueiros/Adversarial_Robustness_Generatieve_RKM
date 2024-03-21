import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from utils import *
import numpy as np
import torch
import torch.nn as nn
import urllib.request
import os.path
from pytorch_msssim import ssim
import pyiqa
import torch.nn.functional as F


# Generate a type 1 attack according to the paper of Sun "TYPE I ATTACK FOR GENERATIVE MODELS" (2020)

""" Instruction for Pre-trained model: Load 'MNIST_trained_RKM_f_mse.tar' when trained with Mean-Squared error for 
    reconstruction. Load (default) 'MNIST_trained_RKM_f_bce.tar' when trained with Binary-Cross entropy error
    for reconstruction """

# Load a Pre-trained model or saved model ====================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', type=str, default='out/MNIST_trained_RKM', help='Enter Filename')
# parser.add_argument('--device', type=str, default='cpu', help='Device type: cuda or cpu')
opt_gen = parser.parse_args()

# sd_mdl = state dictionary model
# sd_mdl = torch.load('{}.tar'.format(opt_gen.filename, map_location=lambda storage, loc: storage)) #lambda storage, loc: storage));  torch.device('cpu')
sd_mdl = torch.load('{}.tar'.format(opt_gen.filename), map_location='cpu')

net1 = sd_mdl['net1'].double().cpu()
net3 = sd_mdl['net3'].double().cpu()
net2 = sd_mdl['net2'].double().cpu()
net4 = sd_mdl['net4'].double().cpu()

net1.load_state_dict(sd_mdl['net1_state_dict'])
net3.load_state_dict(sd_mdl['net3_state_dict'])
net2.load_state_dict(sd_mdl['net2_state_dict'])
net4.load_state_dict(sd_mdl['net4_state_dict'])
h = sd_mdl['h'].double().detach().cpu()
s = sd_mdl['s'].double().detach().cpu()
V = sd_mdl['V'].double().detach().cpu()
U = sd_mdl['U'].double().detach().cpu()

if 'opt' in sd_mdl:
    opt = sd_mdl['opt']
    opt_gen = argparse.Namespace(**vars(opt), **vars(opt_gen))
else:
    opt_gen.mb_size = 200

# print(f"opt_device {opt_gen.device}")

# dimension latent space
dim_h = h.shape[1]

my_lambda = 0.1
nbr_iterations = 100
eta = 0.1  # learning rate

# load the data
opt_gen.shuffle = False
xt, _, _ = get_mnist_dataloader(args=opt_gen)  # loading data without shuffle
xtrain = xt.dataset.train_data[:h.shape[0], :, :, :]  # 60.000, 1, 28, 28; h shape 0: 5000
ytrain = xt.dataset.targets[:h.shape[0], :]

phi_x_mean = net1(xtrain).mean(axis=0)
phi_y_mean = net2(ytrain).mean(axis=0)

##############
# original plot
##############

# plot the first image with his number
nbr_image = 0

xtest = xt.dataset.train_data[nbr_image, :, :, :]
ytest = xt.dataset.targets[nbr_image, :]


# define the loss function to generate the adversarial image
def loss_function(x_ori, x, lam):
    """ Loss function to generate the adversarial image where input different and output same. (type 1 attack) """

    my_loss = torch.linalg.vector_norm(net3(net1(x)) - x_ori) - lam * torch.linalg.vector_norm(
        x - x_ori)  # torch.linalg.vector_norm

    return my_loss


def loss_function_fool(x_ori, x, lam):
    """ Loss function to generate the adversarial image where input similar and output different. (type 2 attack) """

    my_loss = - torch.linalg.vector_norm(net3(net1(x)) - x_ori) + lam * torch.linalg.vector_norm(
        x - x_ori)  # torch.linalg.vector_norm

    return my_loss


#################
# prepare the data
#################

chosen_index = nbr_image

xchosen = xt.dataset.train_data[chosen_index, :, :, :]
image = xchosen.to(opt_gen.device)

image = image[None, :, :, :]
assert image.shape == (1, 1, 28, 28)

image_adv_ori = image.clone().detach()
image_adv = image_adv_ori.clone().detach().requires_grad_(True)

# print(f"requires grad image {image.requires_grad}")
# print(f"requires grad net1 {net1.requires_grad_}")

###############################
# optimize the adversarial image
###############################

print()
print("start iterations")
print()

for i in range(nbr_iterations):
    # print(f"iteration: {i}")
    # print()
    image_adv = image_adv.clone().detach().requires_grad_(True)
    loss_value = loss_function_fool(image_adv_ori, image_adv, my_lambda)
    loss_value.backward()  # 1.0
    # print(image_adv.grad.shape)
    image_adv = image_adv - eta * image_adv.grad
    torch.clamp(image_adv, 0, 1)  # clamp all values between 0 and 1

    """
    for param in my_model.parameters():
        #print(param)
        x_adv_temp = param.clone().detach().numpy().reshape(1, 28, 28)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image[0, 0, :], cmap='Greys_r')
        ax[1].imshow(x_adv_temp[0, :], cmap='Greys_r')
        plt.suptitle("original and transformed image with only the encoder and decoder networks")
        plt.show()
    """

    # loss_array[epoch] = loss.detach().numpy()
    """x_adv_temp = image_adv.reshape(1, 28, 28)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image[0, 0, :], cmap='Greys_r')
    ax[1].imshow(x_adv_temp[0, :], cmap='Greys_r')
    plt.suptitle("original and transformed image with only the encoder and decoder networks")
    plt.show()"""

x_gen_output = net3(net1(image_adv)).clone().detach().reshape(1, 1, 28, 28)
x_gen = image_adv.detach().reshape(1, 1, 28, 28)

image_adv_ori_output = net3(net1(image_adv_ori)).clone().detach().reshape(1, 1, 28, 28)
# print(x_gen)
# print(torch.clamp(x_gen,0,1))

##############
#  Metrics
##############
# SSIM (Structural Similarity Index)
print("SSIM (orig_input - adv_input): ",
      ssim(image, x_gen, data_range=1.0, size_average=True))
print("SSIM (orig_input - orig_output): ",
      ssim(image_adv_ori, image_adv_ori_output, data_range=1.0, size_average=True))
print("SSIM (adv_input - adv_output): ",
      ssim(x_gen, x_gen_output, data_range=1.0, size_average=True))

# LPIPS (lower is better)
lpips = pyiqa.create_metric('lpips', device='cpu')
target_size = (256, 256)  # Adjust the target size as needed
img1_resized = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
img2_resized = F.interpolate(x_gen, size=target_size, mode='bilinear', align_corners=False)
print("LPIPS (orig_input - adv_input): ", lpips(img1_resized.float(), img2_resized.float()))
img3_resized = F.interpolate(image_adv_ori, size=target_size, mode='bilinear', align_corners=False)
img4_resized = F.interpolate(image_adv_ori_output, size=target_size, mode='bilinear', align_corners=False)
print("LPIPS (orig_input - orig_output): ", lpips(img1_resized.float(), img2_resized.float()))
img5_resized = F.interpolate(x_gen_output, size=target_size, mode='bilinear', align_corners=False)
print("LPIPS (adv_input - adv_input): ", lpips(img2_resized.float(), img5_resized.float()))
##############
# Plot the data
##############

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image[0, 0, :], cmap='Greys_r')
ax[1].imshow(x_gen[0, 0, :], cmap='Greys_r')
plt.suptitle("original and transformed image with only the encoder and decoder networks")
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image_adv_ori.reshape(1, 28, 28)[0, :], cmap='Greys_r')
ax[1].imshow(image_adv_ori_output.reshape(1, 28, 28)[0, :], cmap='Greys_r')
plt.suptitle("adversarial image and output of original adversarial image with only the encoder and decoder networks")
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_gen.reshape(1, 28, 28)[0, :], cmap='Greys_r')
ax[1].imshow(x_gen_output.reshape(1, 28, 28)[0, :], cmap='Greys_r')
plt.suptitle("adversarial image and output of adversarial image with only the encoder and decoder networks")
plt.show()

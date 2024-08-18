import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils import *
import numpy as np
import torch
import pyiqa
import torch.nn.functional as F
# from latent import h, U_, V_, s_, net1_, net3_, sd_mdl_

# Generate a type 1 attack according to the paper of Sun "TYPE I ATTACK FOR GENERATIVE MODELS" (2020)

""" Instruction for Pre-trained model: Load 'MNIST_trained_RKM_f_mse.tar' when trained with Mean-Squared error for 
    reconstruction. Load (default) 'MNIST_trained_RKM_f_bce.tar' when trained with Binary-Cross entropy error
    for reconstruction """

"""
CLEAN VERSION
Generate adversarial samples on the genRKM (multi-view but not really used), based on the paper of SUN (type 1 attacks)
Add small number of noise and big change or big noise and small change. Add also one with a target.
GenRKM completely "traversed.

Functions for the attacks and for the image generation. Not really efficient to work with only one image at the time.

When to stop the attack?

Need metric for success.
"""

# Load a Pre-trained model or saved model ====================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--filename', type=str, default='pre_trained_models/MNIST_trained_RKM_f_bce', help='Enter Filename')
parser.add_argument('--filename', type=str, default='out/MNIST_adv_linf_RKM_h_500_WITHlinear', help='Enter Filename')
# parser.add_argument('--device', type=str, default='cpu', help='Device type: cuda or cpu')
# parser.add_argument('--dataset_name', type=str, default='mnist',
#                     help='Dataset name: mnist/fashion-mnist/svhn/dsprites/3dshapes/cars3d')
opt_gen = parser.parse_args()

# sd_mdl = state dictionary model
# sd_mdl = torch.load('{}.tar'.format(opt_gen.filename, map_location=lambda storage, loc: storage)) #lambda storage, loc: storage));  torch.device('cpu')
sd_mdl = torch.load('{}.tar'.format(opt_gen.filename), map_location='cpu')

# .double() added otherwise problem double input and float bias, other solution with to?

net1 = sd_mdl['net1'].float().cpu()
net3 = sd_mdl['net3'].float().cpu()
net2 = sd_mdl['net2'].float().cpu()
net4 = sd_mdl['net4'].float().cpu()

net1.load_state_dict(sd_mdl['net1_state_dict'])
net3.load_state_dict(sd_mdl['net3_state_dict'])
net2.load_state_dict(sd_mdl['net2_state_dict'])
net4.load_state_dict(sd_mdl['net4_state_dict'])
h = sd_mdl['h'].float().detach().cpu()  # N x h_dim
s = sd_mdl['s'].float().detach().cpu()  # N x 1
V = sd_mdl['V'].float().detach().cpu()
U = sd_mdl['U'].float().detach().cpu()  # (64x49) x h_dim
print(U.size())

if 'opt' in sd_mdl:
    opt = sd_mdl['opt']
    opt_gen = argparse.Namespace(**vars(opt), **vars(opt_gen))
else:
    opt_gen.mb_size = 200

# dimension latent space
dim_h = h.shape[1]

# load the data
opt_gen.shuffle = False
xt, _, nChannels = get_dataloader(args=opt_gen)  # loading data without shuffle
xtrain = xt.dataset.train_data[:h.shape[0], :, :, :]  # 60.000, 1, 28, 28; h shape 0: 500
ytrain = xt.dataset.targets[:h.shape[0], :]


phi_x_mean = net1(xtrain).mean(axis=0)
phi_y_mean = net2(ytrain).mean(axis=0)

###########
# Parameters
###########
# select one image from the dataset
# 5000 first images are from the training dataset
nbr_image = 5033
chosen_index_target = 1452  # index of chosen target image for target attack

my_lambda = 0.1  # 10 for type 1 attack, and 0.1 for type 2 and target  #trade-off between similarity input/output
# and difference output/input (depending on attack 1 or 2)
nbr_iterations = 100  # iterations to compute the perturbation
eta = 0.1  # learning rate, proportion of the gradient added to the image

# Similarity input and output
# measured with the Frobenius norm

similarity_input = []
similarity_output = []

##############
# original plot
##############

# plot the selected image

xtest = xt.dataset.train_data[nbr_image, :, :, :]
ytest = xt.dataset.targets[nbr_image, :]

"""
plt.imshow(xtest[0, :, :], cmap = 'Greys_r')

plt.show()

print(f"number of first image {ytest}")
"""


########################################
# Function to compute the latent variable
########################################

# Define and compute Kernel Matrices
def kPCA(X, Y):
    a = torch.mm(X, torch.t(X)) + torch.mm(Y, torch.t(Y))
    nh1 = a.size(0)
    oneN = torch.div(torch.ones(nh1, nh1), nh1).float().to(opt_gen.device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)  # centering
    h, s, _ = torch.svd(a, some=False)
    return h[:, :dim_h], s


def similarity_images(first_image, second_image):
    """
    Measure the similarity between two images using the Frobenius norm
    """

    return float(torch.linalg.vector_norm(first_image - second_image).detach().numpy())


def generate_back(x, U, V, s, h):
    """
    Receives an imput as image, goes to the feature space, then latent space, afteerwards back to the feature space and back to the image space.
    Travels the complete model to generate back the image.
    Input: (1, 1, 28, 28) (first dimension needed for the neural network)
    Output: (1, 28, 28)
    """
    # go to the feature space
    output1 = net1(x)

    # compute (Lambda-V_T*V)^-1
    VT_V = torch.mm(V.T, V)
    Lambda = torch.diag(s[:h.size(1)])
    lam_VTV = torch.linalg.inv(Lambda - VT_V)

    # compute h = (Lambda-V_T*V)^-1*U_T*net1(x)
    # note here the computation is done with everything transposed
    latent1 = torch.mm(torch.mm(output1, U), lam_VTV.T)

    # go back to the feature space U*h
    my_input_net3 = torch.mm(latent1, torch.t(U))

    # go back to the image space
    datax_gen = net3(my_input_net3[0, :])

    # resize the image
    datax_gen = datax_gen.reshape(1, 28, 28)

    return datax_gen


# Algorithm coming from the paper of Sun
# define the loss function to generate the adversarial image

def loss_function(x_ori, x, lam, type1=True):
    """
    Loss function to generate the adversarial image where input different and output same. (type 1 attack)
    Loss function to generate the adversarial image where input similar and output different. (type 2 attack)
    """
    x_back = generate_back(x, U, V, s, h)
    # x_back = net3(net1(x))
    my_loss = torch.linalg.vector_norm(x_back - x_ori) - lam * torch.linalg.vector_norm(
        x - x_ori)  # torch.linalg.vector_norm

    if not type1:
        my_loss = -1 * my_loss  # change sign of the loss function in case type 2 attack

    return my_loss


def loss_function_target(x_ori, x_target, x, lam):
    """ Loss function to generate the adversarial image where input similar and output different and equal to a
    target image. (type 2 attack)"""

    x_back = generate_back(x, U, V, s, h)
    # x_back = net3(net1(x))
    my_loss = torch.linalg.vector_norm(x_back - x_target) + lam * torch.linalg.vector_norm(
        x - x_ori)  # torch.linalg.vector_norm

    return my_loss


def adversarial_attack(adv_image_ori, nbr_it, lam_fun, eta_fun, type1):
    """
    Adversarial attack using algorithm from the paper of Sun (type 1 attack)

    adv_image_ori = original unperturbed image
    adv_image = image that will be perturbed
    nbr_it = number of iterations
    lam_fun = lambda value = trade-off between the two terms
    eta = rate to apply the perturbation on the image, "learning rate"
    type1 = chose between type 1 and type 2 attack
    """

    # initialize the adversarial image that will be perturbed
    adv_image = adv_image_ori.clone().detach().requires_grad_(True)

    # attack on the image
    for i in range(nbr_it):
        adv_image = adv_image.clone().detach().requires_grad_(True)
        loss_value = loss_function(adv_image_ori, adv_image, lam_fun, type1)
        loss_value.backward()  # 1.0
        adv_image = adv_image - eta_fun * adv_image.grad
        torch.clamp(adv_image, 0, 1)  # clamp all values between 0 and 1

        if (i + 1) % 1 == 0:
            similarity_input.append(similarity_images(adv_image_ori, adv_image))
            similarity_output.append(similarity_images(generate_back(adv_image, U, V, s, h), adv_image))

    return adv_image


def adversarial_attack_target(adv_image_ori, x_target, nbr_it, lam_fun, eta_fun):
    """
    Adversarial attack using algorithm from the paper of Sun (type 1 attack)

    adv_image_ori = original unperturbed image
    adv_image = image that will be perturbed
    x_target = target image to which we want that the output of the adversarial image is equal
    nbr_it = number of iterations
    lam_fun = lambda value = trade-off between the two terms
    eta = rate to apply the perturbation on the image, "learning rate"
    """

    # initialize the adversarial image that will be perturbed
    adv_image = torch.tensor(adv_image_ori, requires_grad=True)

    # attack on the image
    # for i in range(nbr_it):
    i = 0
    while torch.norm(adv_image-adv_image_ori) < 3 and i < nbr_it:
        adv_image = adv_image.clone().detach().requires_grad_(True)
        loss_value = loss_function_target(adv_image_ori, x_target, adv_image, lam_fun)
        loss_value.backward()  # 1.0
        adv_image = adv_image - eta_fun * adv_image.grad
        torch.clamp(adv_image, 0, 1)  # clamp all values between 0 and 1

        if (i + 1) % 1 == 0:
            similarity_input.append(similarity_images(adv_image_ori, adv_image))
            similarity_output.append(similarity_images(generate_back(adv_image, U, V, s, h), adv_image))
            # similarity_output.append(similarity_images(net3(net1(adv_image)), adv_image))
            i += 1
    return adv_image


#################
# Prepare the data
#################
datax, datay = xtest.to(opt_gen.device), ytest.to(opt_gen.device)
# transform/project data to feature space

datax_original = datax

image = datax[None, :, :, :]
assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

# starting image for the adversarial attack (equal to the original image)
image_adv_ori = image.clone().detach()
# adversarial image that will be updated
image_adv = image_adv_ori.clone().detach().requires_grad_(True)

print("amount of perturbation", torch.norm(datax_original-image_adv))

# create target image
xchosen_target = xt.dataset.train_data[chosen_index_target, :, :, :]
image_target = xchosen_target.to(opt_gen.device)

image_target = image_target[None, :, :, :]
assert image_target.shape == (1, 1, 28, 28)

############################################
# Generate image back from the original image
############################################

image_gen = generate_back(image, U, V, s, h)
image_gen = image_gen.reshape(1, 28, 28)

##############
# Plot the data
##############


print()
print("Print image")
print()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image[0, 0, :], cmap='Greys_r')
ax[1].imshow(image_target[0, 0, :].detach().numpy(), cmap='Greys_r')
plt.suptitle("Original and transformed image")
plt.show()


# initialize the similarity lists
similarity_input.append(similarity_images(image_adv_ori, image_adv_ori))
similarity_output.append(similarity_images(image_adv_ori, generate_back(image_adv_ori, U, V, s, h)))
# similarity_output.append(similarity_images(image_adv_ori, net3(net1(image_adv_ori))))

# generate the adversarial attack
# image_adv = adversarial_attack(image_adv_ori, nbr_iterations, my_lambda, eta, type1=False)
image_adv = adversarial_attack_target(image_adv_ori, image_target, nbr_iterations, my_lambda, eta)

# output of the adversarial input sample/output of the perturbed image
x_gen_adv_output = generate_back(image_adv, U, V, s, h)
x_gen_adv_output = x_gen_adv_output.clone().detach().reshape(1, 28, 28)
# x_gen_adv_output = net3(net1(image_adv)).clone().detach().reshape(1, 28, 28)
x_gen_adv = image_adv.detach().reshape(1, 28, 28)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(image[0, 0, :], cmap='Greys_r')
ax[0, 1].imshow(image_gen[0, :].detach().numpy(), cmap='Greys_r')
ax[1, 0].imshow(x_gen_adv[0, :], cmap='Greys_r')
ax[1, 1].imshow(x_gen_adv_output[0, :], cmap='Greys_r')
plt.suptitle("Original image, original image back,\n adversarial image and adversarial image back")
plt.show()

print()
# print(similarity_input)
# print(similarity_output)

iteration_array = range(nbr_iterations + 1)

# Plotting the curves with specified colors
plt.plot(iteration_array, similarity_input, color='red', label='Input similarity')
plt.plot(iteration_array, similarity_output, color='blue', label='Output similarity')

# Adding legend
plt.legend()

# Adding labels and title
plt.xlabel('Nbr of iterations')
plt.ylabel('Similarity (Frob Norm)')
plt.title('Similarity in the input and output with original image')

# Display plot
plt.show()

##############
#  Metrics
##############
target_size = (256, 256)  # Adjust the target size as needed
img1_resized = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
img2_resized = F.interpolate(image_gen.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
img3_resized = F.interpolate(x_gen_adv.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
img4_resized = F.interpolate(x_gen_adv_output.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)

# SSIM (Structural Similarity Index)
ssim = pyiqa.create_metric('ssim', device='cpu')
print("SSIM (orig_input - adv_input): ", ssim(img1_resized.float(), img3_resized.float()))
print("SSIM (orig_input - orig_output): ", ssim(img1_resized.float(), img2_resized.float()))
print("SSIM (adv_input - adv_output): ", ssim(img3_resized.float(), img4_resized.float()))
print("SSIM (orig_output - adv_output): ", ssim(img2_resized.float(), img4_resized.float()))

# LPIPS (lower is better)
lpips = pyiqa.create_metric('lpips', device='cpu')
print("LPIPS (orig_input - adv_input): ", lpips(img1_resized.float(), img3_resized.float()))
print("LPIPS (orig_input - orig_output): ", lpips(img1_resized.float(), img2_resized.float()))
print("LPIPS (adv_input - adv_output): ", lpips(img3_resized.float(), img4_resized.float()))
print("LPIPS (orig_output - adv_output): ", lpips(img2_resized.float(), img4_resized.float()))

# Frobenius

###############
# Visualize Perturbation
###############
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(x_gen_adv[0, :] - image[0, 0, :], cmap='Greys_r')
ax[0, 1].imshow(image_gen[0, :].detach() - x_gen_adv_output[0, :], cmap='Greys_r')
ax[1, 0].imshow(image_gen[0, :].detach() - image[0, 0, :], cmap='bwr')
ax[1, 1].imshow(x_gen_adv_output[0, :] - x_gen_adv[0, 0, :], cmap='bwr')

plt.show()

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img3_resized[0, 0, :, :].float() - img1_resized[0, 0, :, :].float(), cmap='Greys_r')
# ax[1].imshow(img4_resized[0, 0, :, :].float() - img2_resized[0, 0, :, :].float(), cmap='Greys_r')
# plt.show()


################
# Disentanglement
################
# b_range = np.arange(-1, 1.1, 0.2)
# c = 100
# n_comp = 10
# fig, ax = plt.subplots(n_comp*2, len(b_range))
# for comp in range(n_comp):
#     for nb, b in enumerate(b_range):
#         h_new = h
#         h_new[:, comp] *= (b*c)
#
#         image_gen = generate_back(image, U, V, s, h_new)
#         image_gen = image_gen.reshape(1, 28, 28)
#
#         image_adv = adversarial_attack(image_adv_ori, nbr_iterations, my_lambda, eta, type1=False)
#         x_gen_adv_output = generate_back(image_adv, U, V, s, h_new)
#         x_gen_adv_output = x_gen_adv_output.clone().detach().reshape(1, 28, 28)
#
#         ax[comp, nb].imshow(image_gen[0, :].detach().numpy(), cmap='Greys_r', interpolation='nearest')
#         ax[comp+n_comp, nb].imshow(x_gen_adv_output[0, :], cmap='Greys_r', interpolation='nearest')
#         ax[comp, nb].axis('off')
#         ax[comp+n_comp, nb].axis('off')
# plt.show()

# 2-D Interpolation %%%%%%%%%%%%

# indx1 = 0
# indx2 = 1
# indx3 = 2
# indx4 = 3
#
# y1 = h[indx1, :]
# y2 = h[indx2, :]
# y3 = h[indx3, :]
# y4 = h[indx4, :]
#
# m = 12
# T, S = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
# T = np.ravel(T, order="F")
# S = np.ravel(S, order="F")
#
# fig5, ax = plt.subplots(m, m)
# it = 0
# for i in range(m):
#     for j in range(m):
#         # weights
#         lambd = np.flip(
#             np.hstack((S[it] * T[it], (1 - S[it]) * T[it], S[it] * (1 - T[it]), (1 - S[it]) * (1 - T[it]))), 0)
#
#         # computation
#         yop = lambd[0] * y1 + lambd[1] * y2 + lambd[2] * y3 + lambd[3] * y4
#
#         x_gen = net3(torch.mv(U, yop)).detach().numpy()
#
#         x_gen = x_gen.reshape(1, 28, 28)
#         y_gen = net4(torch.mv(V, yop)).detach().numpy()
#
#         y_gen[y_gen < 0] = 0
#         ps = np.exp(y_gen)
#         ps /= np.sum(ps)
#         ind = np.argpartition(ps, -2)[-2:]
#
#         image_adv = adversarial_attack(image_adv_ori, nbr_iterations, my_lambda, eta, type1=False)
#         U_star = torch.mv(U, yop).unsqueeze(1).expand(-1, dim_h)
#         V_star = torch.mv(V, yop).unsqueeze(1).expand(-1, dim_h)
#         x_gen_adv_output = generate_back(image_adv, U_star, V_star, s, h)
#         x_gen_adv_output = x_gen_adv_output.clone().detach().reshape(1, 28, 28)
#
#         # ax[i, j].imshow(x_gen[0, :], cmap='Greys_r')
#         # ax[i, j].imshow(x_gen_adv_output[0, :], cmap='Greys_r')
#         ax[i, j].imshow(x_gen_adv_output[0, :] - x_gen[0, :], cmap='bwr')
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
#         it += 1
# plt.suptitle('2D Interpolation')
# plt.show()

#######################
# Components analysis
#######################

# Robust: 1, 4
# Not-robust: 2, 0

# def components_analysis.png(nbr_image=0, robust_comp=(1,4), nonrobust_comp=(2, 0)):


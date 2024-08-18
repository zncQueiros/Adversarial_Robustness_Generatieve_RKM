import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from utils import *
import numpy as np
import torch

sd_mdl = torch.load('pre_trained_models/MNIST_trained_RKM_f_bce.tar', map_location=torch.device('cpu'))

net1 = sd_mdl['net1'].float().cpu()
net3 = sd_mdl['net3'].float().cpu()
net2 = sd_mdl['net2'].float().cpu()
net4 = sd_mdl['net4'].float().cpu()
net1.load_state_dict(sd_mdl['net1_state_dict'])
net3.load_state_dict(sd_mdl['net3_state_dict'])
net2.load_state_dict(sd_mdl['net2_state_dict'])
net4.load_state_dict(sd_mdl['net4_state_dict'])
h = sd_mdl['h'].float().detach().cpu()
s = sd_mdl['s'].float().detach().cpu()
V = sd_mdl['V'].float().detach().cpu()
U = sd_mdl['U'].float().detach().cpu()

sd_mdl_ = torch.load('out/MNIST_adv_linf_RKM_h_500_withlinear.tar', map_location=torch.device('cpu'))

net1_ = sd_mdl_['net1'].float().cpu()
net3_ = sd_mdl_['net3'].float().cpu()
net2_ = sd_mdl_['net2'].float().cpu()
net4_ = sd_mdl_['net4'].float().cpu()
net1_.load_state_dict(sd_mdl['net1_state_dict'])
net3_.load_state_dict(sd_mdl['net3_state_dict'])
net2_.load_state_dict(sd_mdl['net2_state_dict'])
net4_.load_state_dict(sd_mdl['net4_state_dict'])
h_ = sd_mdl_['h'].float().detach().cpu()
s_ = sd_mdl_['s'].float().detach().cpu()
V_ = sd_mdl_['V'].float().detach().cpu()
U_ = sd_mdl_['U'].float().detach().cpu()


def euclidean_distance(vec1, vec2):
    return np.min([np.linalg.norm(vec1 - vec2), np.linalg.norm(vec1 + vec2)])

def cosine_similarity(vec1, vec2):
    dot_product = np.max([np.dot(vec1, vec2), np.dot(vec1, -vec2)])
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

euc_dist = []
cos_sim = []

for col1, col2 in zip(torch.t(h), torch.t(h_)):
    euclidean_dist = euclidean_distance(col1, col2)
    cosine_sim = cosine_similarity(col1, col2)
    euc_dist.append(euclidean_dist)
    cos_sim.append(cosine_sim)
#     print(f"Euclidean Distance: {euclidean_dist}, Cosine Similarity: {cosine_sim}")
#
print(np.argsort(euc_dist))
print(np.argsort(cos_sim))

# print(np.abs(np.argsort(cos_sim) - np.arange(500)).mean())
# print(np.sort(cos_sim)[::-1])

# method 1: multiply a constant on 'robust eigenvectors'
# top_m = np.argsort(euc_dist)[:30]
# h[:, top_m] *= 10

# method 2: replace the first n` vectors in h with h`
# n = 100
# h[:, :n] = h_[:, :n] * 10
# s[:n] = s_[:n]

# comp = (1, 137, 19, 469, 388, 107)
comp = (1, 19, 4, 57)
# print(h.var(dim=0))
nbr_image = 5
m = 11
perturbation = np.linspace(-3, 3, m)
fig1, ax = plt.subplots(m, len(comp))
for i in range(m):
    for j in range(len(comp)):
        h_try = copy.deepcopy(h)
        h_try[nbr_image, comp[j]] += perturbation[i] * h[:, comp[j]].std()
        # x_gen_base = net3(torch.mv(U, h[nbr_image, :])).detach().numpy()
        print(U.shape, h_try[nbr_image, :].shape)
        x_gen_try = net3(torch.mv(U, h_try[nbr_image, :])).detach().numpy()

        ax[i, j].imshow(x_gen_try[0, 0, :], cmap='Greys_r')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.show()

nbr_comp = 200
index1 = np.arange(nbr_comp)
index2 = np.argsort(cos_sim)[:nbr_comp]
index3 = np.argsort(cos_sim)[::-1][:nbr_comp]
idx_list = [index1, index2, index3]

fig2, ax = plt.subplots(1, 3)
for k in range(3):
    h_try = h.clone()
    U_try = U.clone()
    idx = idx_list[k]

    U_try[:, np.setdiff1d(np.arange(U_try.shape[1]), idx)] = 0
    h_try[:, np.setdiff1d(np.arange(h_try.shape[1]), idx)] = 0
    x_gen_try = net3(torch.mv(U[:, :], h_try[nbr_image, :])).cpu().detach().numpy()
    ax[k].imshow(x_gen_try[0, 0, :], cmap='Greys_r')
    ax[k].set_xticks([])
    ax[k].set_yticks([])
plt.show()


# print(h.var(dim=0))
fig3, ax = plt.subplots(m, len(comp))
for i in range(m):
    for j in range(len(comp)):
        h_try = h.clone()
        U_try = U.clone()
        idx = index2
        U_try[:, np.setdiff1d(np.arange(U_try.shape[1]), idx)] = 0
        h_try[:, np.setdiff1d(np.arange(h_try.shape[1]), idx)] = 0
        h_try[nbr_image, comp[j]] += perturbation[i] * h[:, comp[j]].std()
        # x_gen_base = net3(torch.mv(U, h[nbr_image, :])).detach().numpy()
        x_gen_try = net3(torch.mv(U, h_try[nbr_image, :])).detach().numpy()

        ax[i, j].imshow(x_gen_try[0, 0, :], cmap='Greys_r')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.show()

# comp = ()
fig4, ax = plt.subplots(m, len(comp))
for i in range(m):
    for j in range(len(comp)):
        h_try = h.clone()
        U_try = U.clone()
        idx = index1
        U_try[:, np.setdiff1d(np.arange(U_try.shape[1]), idx)] = 0
        h_try[:, np.setdiff1d(np.arange(h_try.shape[1]), idx)] = 0
        h_try[nbr_image, comp[j]] += perturbation[i] * h[:, comp[j]].std()
        # x_gen_base = net3(torch.mv(U, h[nbr_image, :])).detach().numpy()
        x_gen_try = net3(torch.mv(U, h_try[nbr_image, :])).detach().numpy()

        ax[i, j].imshow(x_gen_try[0, 0, :], cmap='Greys_r')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.show()
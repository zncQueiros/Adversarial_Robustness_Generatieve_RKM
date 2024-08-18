import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

from dataloader import get_dataloader

device = torch.device("cpu")

h_dim = 10
torch.nn.Module.dump_patches = True


class create_dirs:
    """ Creates directories for Checkpoints and saving trained models """

    def __init__(self, ct):
        self.ct = ct
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = 'Mul_trained_RKM_{}.tar'.format(self.ct)

    def create(self):
        if not os.path.exists('cp/'):
            os.makedirs('cp/')

        if not os.path.exists('out/'):
            os.makedirs('out/')

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}'.format(self.dircp))


def rbf_kernel(X, sigma):
    sq_dists = torch.cdist(X, X, p=2) ** 2
    return torch.exp(-sq_dists / (2 * sigma ** 2))


def linear_kernel(X):
    return X @ X.T


def polynomial_kernel(X, degree=3, gamma=1, coef0=1):
    return (gamma * X @ X.T + coef0) ** degree


def sigmoid_kernel(X, gamma=1, coef0=1):
    return torch.tanh(gamma * X @ X.T + coef0)


# class PKPCA:
#     def __init__(self, n_components, sigma=1.0, max_iter=100):
#         self.n_components = n_components
#         self.sigma = sigma
#         self.max_iter = max_iter
#
#     def _compute_kernel(self, X):
#         # sigma_kernel = 1.0
#         # sq_dists = torch.cdist(X, X, p=2) ** 2
#         # K = torch.exp(-sq_dists / (2 * sigma_kernel ** 2))
#         K = X @ X.T
#         return K
#
#     def fit_transform(self, X):
#         N, D = X.size()
#         K = self._compute_kernel(X)
#
#         # Center the kernel matrix
#         oneN = torch.ones(N, N, device=K.device) / N
#         K_centered = K - oneN @ K - K @ oneN + oneN @ K @ oneN
#
#         # Initialize latent variables
#         W_old = torch.rand(N, self.n_components, device=X.device)
#         W_new = torch.empty(N, self.n_components, device=X.device)  # N*M
#         RMSE = mean_squared_error(W_old.cpu(), W_new.cpu()) ** 0.5
#
#         # --- EM Algorithm:
#
#         if self.sigma == 0:
#             W_all = torch.zeros(N, self.n_components, device=X.device)
#             for _ in range(self.max_iter):
#                 RMSEdiff = 1
#                 while RMSEdiff > 1e-7:
#                     RMSEold = RMSE
#                     Omega = torch.inverse(W_old.T @ W_old) @ (W_old.T @ K_centered)  # E step
#                     W_new = (K_centered @ Omega.T) @ torch.inverse(Omega @ Omega.T)  # M step
#                     MSE = torch.mean((W_old - W_new) ** 2)
#                     RMSE = torch.sqrt(MSE)
#                     W_old = W_new
#                     RMSEdiff = abs(RMSE - RMSEold)
#                 W_all += W_new
#         else:
#             W_all = torch.zeros(N, self.n_components, device=X.device)
#             for _ in range(self.max_iter):
#                 sigma_new = torch.rand(1).item()
#                 dif_sigma = 1
#                 RMSEdiff = 1
#                 while RMSEdiff > 1e-7 or dif_sigma > 1e-8:
#                     RMSEold = RMSE
#                     M = W_old.T @ W_old + self.sigma * torch.eye(self.n_components, device=X.device)  # M*M
#                     E_Zn = torch.inverse(M) @ W_old.T @ K_centered  # M*N
#                     E_Zn_ZnT = self.sigma * torch.inverse(M) + E_Zn @ E_Zn.T  # M*M
#                     W_new = K_centered @ E_Zn.T @ torch.inverse(E_Zn_ZnT)  # N*M
#                     sigma_sq_new = 0
#                     for i in range(N):
#                         trace = torch.trace(W_new @ E_Zn_ZnT @ W_new.T)
#                         a_01 = torch.norm(K_centered[:, i]) ** 2
#                         a_02 = E_Zn[:, i].T @ W_new.T @ K_centered[:, i]
#                         sigma_sq_new += a_01 - 2 * a_02 + trace
#                     sigma_sq_new /= (N * D)
#                     MSE = torch.mean((W_old - W_new) ** 2)
#                     RMSE = torch.sqrt(MSE)
#                     dif_sigma = abs(self.sigma - sigma_sq_new)
#                     W_old = W_new
#                     self.sigma = sigma_sq_new
#                     RMSEdiff = abs(RMSE - RMSEold)
#                 W_all += W_new
#         W_mean = W_all / self.max_iter
#
#         U, S, V = torch.svd(W_mean)
#         h = U.T @ K_centered
#
#         return h, S
#

def adv_train(X, model, adversary):
    """
    Adversarial training. Returns pertubed mini batch.
    """

    # If adversarial training, need a snapshot of
    # the model at each batch to compute grad, so
    # as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    adversary.model = model_cp

    X_adv = adversary.perturb(X)

    return X_adv


def final_compute(args, net1, kPCA):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _ = get_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))  # 3136x128 without linear layer
    h, s = kPCA(xtr)
    return torch.mm(torch.t(xtr), h), h, s


def final_compute2(args, net1, kPCA):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _ = get_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))  # 3136x128 without linear layer
    h, s = kPCA(xtr)
    return torch.mm(torch.t(xtr), h), h, s


def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(Variable(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


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
        self.fc1 = torch.nn.Linear(c * 2 * 7 * 7, x_fdim)  # 3136 x 128

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
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        # x = F.leaky_relu(self.fc1(x), negative_slope=0.2)  # Not
        # x = x.view(1, capacity * 2, 7, 7)
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

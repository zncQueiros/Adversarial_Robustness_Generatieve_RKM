import time
from datetime import datetime
import math
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch

from utils import *
from attack import LinfPGDAttack, FGSMAttack

os.environ["TORCH_METAL_ENABLED"] = "1"
torch.autograd.set_detect_anomaly(False)
# Model Settings =================================================================================================
parser = argparse.ArgumentParser(description='Gen-RKM Model')

parser.add_argument('--dataset_name', type=str, default='mnist',
                    help='Dataset name: mnist/fashion-mnist/svhn/dsprites/3dshapes/cars3d')
parser.add_argument('--N', type=int, default=5000, help='Total # of samples')
parser.add_argument('--mb_size', type=int, default=100, help='Mini-batch size. See utils.py')
parser.add_argument('--h_dim', type=int, default=128, help='Dim of latent vector')
parser.add_argument('--capacity', type=int, default=32, help='Capacity of network. See utils.py')
parser.add_argument('--x_fdim', type=int, default=128, help='Input x_fdim. See utils.py')
parser.add_argument('--y_fdim', type=int, default=20, help='Input y_fdim. See utils.py')
parser.add_argument('--c_accu', type=float, default=100, help='Input weight on recons_error')

# Training Settings =============================
parser.add_argument('--lr', type=float, default=1e-4, help='Input learning rate for optimizer')
parser.add_argument('--max_epochs', type=int, default=1, help='Input max_epoch for cut-off')
parser.add_argument('--delay', type=int, default=10, help='Start adversarial training after delay')
parser.add_argument('--device', type=str, default='mps', help='Device type: cuda|cpu|mps')
parser.add_argument('--workers', type=int, default=0, help='# of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: true or false')
parser.add_argument('--adv', type=str, default=True, help='adv|cln: adv training or not')

opt = parser.parse_args()
# ==================================================================================================================

# load raw data===============================================================================
ct = time.strftime("%Y%m%d-%H%M")
dirs = create_dirs(opt.dataset_name)
dirs.create()

# Load Training Data
xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

train_loss = []
validation_loss = []


# Define and compute Kernel Matrices
def kPCA(X):
    a = sigmoid_kernel(X)
    # a = torch.mm(X, torch.t(X))
    nh1 = a.size(0)
    oneN = torch.div(torch.ones(nh1, nh1), nh1).float().to(opt.device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)  # centering
    h, s, _ = torch.svd(a, some=False)
    return h[:, :opt.h_dim], s


class PKPCA:
    def __init__(self, n_components, sigma=1.0, max_iter=100):
        self.n_components = n_components
        self.sigma = sigma
        self.max_iter = max_iter

    def _compute_kernel(self, X):
        # sigma_kernel = 1.0
        # sq_dists = torch.cdist(X, X, p=2) ** 2
        # K = torch.exp(-sq_dists / (2 * sigma_kernel ** 2))
        K = X @ X.T
        return K

    def fit_transform(self, X):
        N, D = X.size()
        K = self._compute_kernel(X)

        # Center the kernel matrix
        oneN = torch.ones(N, N, device=K.device) / N
        K_centered = K - oneN @ K - K @ oneN + oneN @ K @ oneN

        # Initialize latent variables
        W_old = torch.rand(N, self.n_components, device=X.device)
        W_new = torch.empty(N, self.n_components, device=X.device)  # N*M
        RMSE = mean_squared_error(W_old.cpu().detach().numpy(), W_new.cpu().detach().numpy()) ** 0.5

        # --- EM Algorithm:

        if self.sigma == 0:
            W_all = torch.zeros(N, self.n_components, device=X.device)
            for _ in range(self.max_iter):
                RMSEdiff = 1
                while RMSEdiff > 1e-7:
                    RMSEold = RMSE
                    Omega = torch.inverse(W_old.T @ W_old) @ (W_old.T @ K_centered)  # E step
                    W_new = (K_centered @ Omega.T) @ torch.inverse(Omega @ Omega.T)  # M step
                    MSE = torch.mean((W_old - W_new) ** 2)
                    RMSE = torch.sqrt(MSE)
                    W_old = W_new
                    RMSEdiff = abs(RMSE - RMSEold)
                W_all += W_new
        else:
            W_all = torch.zeros(N, self.n_components, device=X.device)
            for _ in range(self.max_iter):
                sigma_new = torch.rand(1).item()
                dif_sigma = 1
                RMSEdiff = 1
                while RMSEdiff > 1e-7 or dif_sigma > 1e-8:
                    RMSEold = RMSE
                    M = W_old.T @ W_old + self.sigma * torch.eye(self.n_components, device=X.device)  # M*M
                    E_Zn = torch.inverse(M) @ W_old.T @ K_centered  # M*N
                    E_Zn_ZnT = self.sigma * torch.inverse(M) + E_Zn @ E_Zn.T  # M*M
                    W_new = K_centered @ E_Zn.T @ torch.inverse(E_Zn_ZnT)  # N*M
                    for i in range(N):
                        trace = torch.trace(W_new @ E_Zn_ZnT @ W_new.T)
                        a_01 = torch.norm(K_centered[:, i]) ** 2
                        a_02 = E_Zn[:, i].T @ W_new.T @ K_centered[:, i]
                        sigma_new += a_01 - 2 * a_02 + trace
                    sigma_new /= (N * D)
                    MSE = torch.mean((W_old - W_new) ** 2)
                    RMSE = torch.sqrt(MSE)
                    dif_sigma = abs(self.sigma - sigma_new)
                    W_old = W_new
                    self.sigma = sigma_new
                    RMSEdiff = abs(RMSE - RMSEold)
                W_all += W_new
        W_mean = W_all / self.max_iter

        U, S, V = torch.svd(W_mean)
        h = U.T @ K_centered

        return h, S


# Energy function
def rkm_loss(output1, X):
    h, s = kPCA(output1)
    # pkpca = PKPCA(n_components=opt.h_dim, sigma=1)
    # h, s = pkpca.fit_transform(output1)
    U = torch.mm(torch.t(output1), h)

    x_tilde = net3(torch.mm(h, torch.t(U)))

    # Costs
    f1 = torch.trace(torch.mm(torch.mm(output1, U), torch.t(h)))
    f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(torch.diag(s[:opt.h_dim]), torch.t(h))))
    f3 = 0.5 * (torch.trace(torch.mm(torch.t(U), U)))
    recon_loss = torch.nn.MSELoss()
    f4 = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim))  # reconstruction loss

    loss = - f1 + f3 + f2 + 0.5 * (- f1 + f3 + f2) ** 2 + opt.c_accu * f4  # stabilized loss
    return loss


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        output = self.Encoder(x)
        h, s = kPCA(output)
        # pkpca = PKPCA(n_components=opt.h_dim)
        # h, s = pkpca.fit_transform(output1)
        U = torch.mm(torch.t(output), h)

        x_tilde = self.Decoder(torch.mm(h, torch.t(U)))

        return x_tilde


# Initialize Feature/Pre-image maps
net1 = Net1().float().to(opt.device)
net2 = Net2().float().to(opt.device)
net3 = Net3().float().to(opt.device)
net4 = Net4().float().to(opt.device)

# params = list(net1.parameters()) + list(net3.parameters()) + list(net2.parameters()) + list(net4.parameters())
params = list(net1.parameters()) + list(net3.parameters())
optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=0)
model = Model(Encoder=net1, Decoder=net3)

# Adversarial attack
# adversary = FGSMAttack(epsilon=0.3)
adversary = LinfPGDAttack(device=opt.device, random_start=False, amount=30)

# Train ===============================================================================
l_cost = 6  # Costs from where checkpoints will be saved
t = 1
cost = np.inf  # Initialize cost
start = datetime.now()
while cost > 0.2 and t <= opt.max_epochs:  # run epochs until convergence
    avg_loss = 0
    avg_loss_val = 0
    for i, (datax, datay) in enumerate(xtrain):
        if i < math.ceil(opt.N / opt.mb_size):
            datax = datax.to(opt.device)
            output1 = net1(datax)
            loss = rkm_loss(output1, datax)

            # adversarial training
            if t > opt.delay and opt.adv:
                x_adv = adv_train(datax, model, adversary).to(opt.device)
                output1_adv = net1(x_adv)
                loss_adv = rkm_loss(output1_adv, x_adv)
                loss = (loss + loss_adv) / 2

                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(datax[0, 0, :].cpu(), cmap='Greys_r', vmin=0, vmax=1)
                # ax[1].imshow(x_adv[0, 0, :].cpu().detach().numpy(), cmap='Greys_r', vmin=0, vmax=1)
                # plt.suptitle("Original and transformed image")
                # plt.show()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().numpy()

        elif i < math.ceil(opt.N / opt.mb_size) + 10:  # 10 minibatch size for the validation
            datax = datax.to(opt.device)
            output1 = net1(datax)
            loss = rkm_loss(output1, datax)
            if t > 10 and opt.adv:
                y_pred = pred_batch(datax, net1).to(opt.device)
                x_adv = adv_train(datax, model, adversary).to(opt.device)
                output1_adv = net1(x_adv)
                # output2 = net2(datay)
                loss_adv = rkm_loss(output1_adv, datax)
                loss = (loss + loss_adv) / 2
            avg_loss_val += loss.detach().cpu().numpy()

        else:
            break
    cost = avg_loss

    train_loss.append(avg_loss)
    validation_loss.append(avg_loss_val)

    # Remember the lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)

    dirs.save_checkpoint({
        'epochs': t + 1,
        'net1_state_dict': net1.state_dict(),
        'net3_state_dict': net3.state_dict(),
        'net2_state_dict': net2.state_dict(),
        'net4_state_dict': net4.state_dict(),
        'l_cost': l_cost,
        'optimizer': optimizer.state_dict()}, is_best)
    print(t, cost)
    t += 1
print('Finished Training in: {}. Lowest cost: {}'.format(str(datetime.now() - start), l_cost))

if os.path.exists('cp/{}'.format(dirs.dircp)):
    sd_mdl = torch.load('cp/{}'.format(dirs.dircp))
    net1.load_state_dict(sd_mdl['net1_state_dict'])
    net3.load_state_dict(sd_mdl['net3_state_dict'])
    net2.load_state_dict(sd_mdl['net2_state_dict'])
    net4.load_state_dict(sd_mdl['net4_state_dict'])

U, h, s = final_compute(opt, net1, kPCA)
print(U.size())

# Save Model =============
torch.save({'net1': net1,
            'net3': net3,
            'net2': net2,
            'net4': net4,
            'net1_state_dict': net1.state_dict(),
            'net3_state_dict': net3.state_dict(),
            'net2_state_dict': net2.state_dict(),
            'net4_state_dict': net4.state_dict(),
            'h': h,
            'opt': opt, 's': s, 'U': U, 'V': V}, 'out/{}'.format(dirs.dirout))

# Plot the curves
plt.semilogy(range(len(train_loss)), train_loss, color='red', label='Training loss')
plt.semilogy(range(len(validation_loss)), validation_loss, color='blue', label='Validation loss')

# Add labels and title
plt.xlabel('Number of epochs')
plt.ylabel('Loss (log-scale)')
plt.title('Training and validation loss for the training of the adversarial genRKM')

# Add legend
plt.legend()

# Show the plot
#plt.grid(True)
plt.show()

# Plot the curves
plt.semilogy(range(len(train_loss)), train_loss, color='red', label='Training loss')
plt.semilogy(range(len(validation_loss)), validation_loss, color='blue', label='Validation loss')

# Add labels and title
plt.xlabel('Number of epochs')
plt.ylabel('Loss (log-scale)')
plt.title('Training and validation loss for the training of the adversarial genRKM')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

train_loss_array = np.array(train_loss)
validation_loss_array = np.array(validation_loss)

np.save('Training_curves/train_curve_advgeRKM_200epo_noLinLay.npy', train_loss_array)
np.save('Training_curves/validation_curve_advgeRKM_200epo_noLinLay.npy', validation_loss_array)


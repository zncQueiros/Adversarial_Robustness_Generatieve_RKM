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
parser.add_argument('--c_accu', type=float, default=100, help='Input weight on recons_error')

# Training Settings =============================
parser.add_argument('--lr', type=float, default=1e-4, help='Input learning rate for optimizer')
parser.add_argument('--max_epochs', type=int, default=200, help='Input max_epoch for cut-off')
parser.add_argument('--delay', type=int, default=10, help='Start adversarial training after delay')
parser.add_argument('--device', type=str, default='mps', help='Device type: cuda|cpu|mps')
parser.add_argument('--workers', type=int, default=0, help='# of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: true or false')
parser.add_argument('--adv', type=str, default=True, help='adv|cln: adv training or not')
parser.add_argument('--pert', type=float, default=30, help='the amount of perturbation')

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
    a = torch.mm(X, torch.t(X))
    nh1 = a.size(0)
    oneN = torch.div(torch.ones(nh1, nh1), nh1).float().to(opt.device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)  # centering
    h, s, _ = torch.linalg.svd(a, full_matrices=False)
    return h[:, :opt.h_dim], s


# Energy function
def rkm_loss(output1, X):
    h, s = kPCA(output1)
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
        U = torch.mm(torch.t(output), h)

        x_tilde = self.Decoder(torch.mm(h, torch.t(U)))

        return x_tilde


# Initialize Feature/Pre-image maps
net1 = Net1().float().to(opt.device)
net3 = Net3().float().to(opt.device)

params = list(net1.parameters()) + list(net3.parameters())
optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=0)
model = Model(Encoder=net1, Decoder=net3)

# Adversarial attack
adversary = LinfPGDAttack(device=opt.device, random_start=False, amount=opt.pert)

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
        'l_cost': l_cost,
        'optimizer': optimizer.state_dict()}, is_best)
    print(t, cost)
    t += 1
print('Finished Training in: {}. Lowest cost: {}'.format(str(datetime.now() - start), l_cost))

if os.path.exists('cp/{}'.format(dirs.dircp)):
    sd_mdl = torch.load('cp/{}'.format(dirs.dircp))
    net1.load_state_dict(sd_mdl['net1_state_dict'])
    net3.load_state_dict(sd_mdl['net3_state_dict'])

U, h, s = final_compute(opt, net1, kPCA)

# Save Model =============
torch.save({'net1': net1,
            'net3': net3,
            'net1_state_dict': net1.state_dict(),
            'net3_state_dict': net3.state_dict(),
            'h': h,
            'opt': opt, 's': s, 'U': U}, 'out/{}'.format(dirs.dirout))

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

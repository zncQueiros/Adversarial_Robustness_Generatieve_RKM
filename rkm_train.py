import copy
import os
from datetime import datetime
import numpy as np
import math
import argparse
from utils import *
from dataloader import get_mnist_dataloader
from model import *
from attack import LinfPGDAttack, FGSMAttack


torch.autograd.set_detect_anomaly(False)
# Model Settings =================================================================================================
parser = argparse.ArgumentParser(description='Gen-RKM Model')

parser.add_argument('--N', type=int, default=1000, help='Total # of samples')
parser.add_argument('--mb_size', type=int, default=100, help='Mini-batch size. See utils.py')
parser.add_argument('--h_dim', type=int, default=10, help='Dim of latent vector')
parser.add_argument('--capacity', type=int, default=32, help='Capacity of network. See utils.py')
parser.add_argument('--x_fdim', type=int, default=128, help='Input x_fdim. See utils.py')
parser.add_argument('--y_fdim', type=int, default=20, help='Input y_fdim. See utils.py')
parser.add_argument('--c_accu', type=float, default=100, help='Input weight on recons_error')

# Training Settings =============================
parser.add_argument('--lr', type=float, default=1e-4, help='Input learning rate for optimizer')
parser.add_argument('--max_epochs', type=int, default=100, help='Input max_epoch for cut-off')
parser.add_argument('--delay', type=int, default=10, help='Start adversarial training after delay')
parser.add_argument('--device', type=str, default='cpu', help='Device type: cuda or cpu')
parser.add_argument('--workers', type=int, default=0, help='# of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: true or false')

opt = parser.parse_args()
# ==================================================================================================================

# load raw data===============================================================================
# Load Training Data
xtrain, ipVec_dim, nChannels = get_mnist_dataloader(args=opt)

ct = time.strftime("%Y%m%d-%H%M")
dirs = create_dirs(ct=ct)
dirs.create()

# Define and compute Kernel Matrices
def kPCA(X, Y):
    a = torch.mm(X, torch.t(X)) + torch.mm(Y, torch.t(Y))
    nh1 = a.size(0)
    oneN = torch.div(torch.ones(nh1, nh1), nh1).double().to(opt.device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)  # centering
    h, s, _ = torch.svd(a, some=False)
    return h[:, :opt.h_dim], s


# Energy function
def rkm_loss(output1, X, output2, Y):
    h, s = kPCA(output1, output2)
    U = torch.mm(torch.t(output1), h)
    V = torch.mm(torch.t(output2), h)

    x_tilde = net3(torch.mm(h, torch.t(U)))
    y_tilde = net4(torch.mm(h, torch.t(V)))

    # Costs
    f1 = torch.trace(torch.mm(torch.mm(output1, U), torch.t(h))) + torch.trace(
        torch.mm(torch.mm(output2, V), torch.t(h)))
    f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(torch.diag(s[:opt.h_dim]), torch.t(h))))
    f3 = 0.5 * ((torch.trace(torch.mm(torch.t(U), U))) + (torch.trace(torch.mm(torch.t(V), V))))
    recon_loss = torch.nn.MSELoss()
    f4 = recon_loss(x_tilde.view(-1, ipVec_dim), X.view(-1, ipVec_dim)) + recon_loss(y_tilde.view(-1, 10), Y.view(-1,
                                                                                                                  10))  # reconstruction loss

    loss = - f1 + f3 + f2 + 0.5 * (- f1 + f3 + f2) ** 2 + opt.c_accu * f4  # stabilized loss
    return loss


# Initialize Feature/Pre-image maps
net1 = Net1().double().to(opt.device)
net2 = Net2().double().to(opt.device)
net3 = Net3().double().to(opt.device)
net4 = Net4().double().to(opt.device)

params = list(net1.parameters()) + list(net3.parameters()) + list(net2.parameters()) + list(net4.parameters())
optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=0)

# Adversarial attack
# adversary = FGSMAttack(epsilon=0.3)
# adversary = LinfPGDAttack()

# Train ===============================================================================
l_cost = 6  # Costs from where checkpoints will be saved
t = 1
cost = np.inf  # Initialize cost
latent_adversarial_training = True
start = datetime.now()
while cost > 0.2 and t <= opt.max_epochs:  # run epochs until convergence
    avg_loss = 0
    for i, (datax, datay) in enumerate(xtrain):
        if i < math.ceil(opt.N / opt.mb_size):
            datax, datay = datax.to(opt.device), datay.to(opt.device)
            output1 = net1(datax)
            output2 = net2(datay)
            loss = rkm_loss(output1, datax, output2, datay)

            # # adversarial training
            # if i+1 > opt.delay and not latent_adversarial_training:
            #     y_pred = pred_batch(datax, net1)
            #     x_adv = adv_train(datax, y_pred, net1, rkm_loss, adversary)
            #     output1_adv = net1(x_adv)
            #     loss_adv = rkm_loss(output1_adv, x_adv, output2, datay)
            #     loss = (loss + loss_adv) / 2
            #
            # # latent space adversarial training
            # if i+1 > opt.delay and latent_adversarial_training:
            #     y_pred = pred_batch(datax, net1)
            #     output1_adv = adv_train(output1, y_pred, copy.deepcopy, rkm_loss, adversary)
            #     loss_adv = rkm_loss(output1_adv, datax, output2, datay)
            #     loss = (loss + loss_adv) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().numpy()
        else:
            break
    cost = avg_loss

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

U, V, h, s = final_compute(opt, net1, net2, kPCA)

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

import torch
import os
import time
from model import net3

device = torch.device("cpu")

h_dim = 10

# Define centered Kernel matrix
def kPCA(X):
    a = torch.mm(X, torch.t(X))
    nh1 = X.size(0)
    oneN = torch.div(torch.ones(nh1, nh1), nh1).to(device)
    return a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)  # kernel centering
    # da = torch.diag(1 / torch.sqrt(torch.diag(a)))
    # a = torch.mm(torch.mm(da, a), da)
    # print('condition number: ', np.linalg.cond(a.cpu().detach().numpy()))


# Define initial loss
def h1_loss(op1, X):
    a = kPCA(op1)
    h1, s1, _ = torch.linalg.svd(a)  # we want full matrix
    h1 = h1[:, : h_dim]
    U = torch.mm(torch.t(op1), h1)

    x_tilde = net3(torch.mm(h1, torch.t(U)))

    # Cost
    f1 = torch.trace(torch.mm(torch.mm(op1, U), torch.t(h1)))
    f2 = 0.5 * s1[0] * torch.trace(torch.mm(h1, torch.t(h1)))
    f3 = 0.5 * (torch.trace(torch.mm(torch.t(U), U)))
    recon_loss2 = torch.nn.MSELoss(reduction='sum')
    f4 = recon_loss2(x_tilde.view(-1, 784), X.view(-1, 784)) / X.size(0)

    loss = - f1 + f3 + f2 + 0.5 * (- f1 + f2 + f3) ** 2 + 100 * f4  # Stabilized loss
    return loss, f4, h1, s1, U, a


ct = time.strftime("%Y%m%d-%H%M")
dircp = 'checkpoint.pth_{}.tar'.format(ct)

if not os.path.exists('cp/'):
    os.makedirs('cp/')


def save_checkpoint(state, is_best, filename='cp/{}'.format(dircp)):
    if is_best:
        torch.save(state, filename)

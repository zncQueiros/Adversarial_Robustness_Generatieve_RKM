import os
import time
import numpy as np
from utils import *
from dataloader import get_mnist
from model import *


torch.autograd.set_detect_anomaly(False)

# load raw data===============================================================================
xtrain = get_mnist()[0]
xtrain = xtrain[:N, :, :, :]

# Add noise to 20% data
xtrain[-int(N * 0.2):, :, :, :] = xtrain[-int(N * 0.2):, :, :, :] \
                                  + 0.5 * torch.randn((int(N * 0.2), 1, 28, 28)) + 0.5
xtrain = xtrain.to(device)

params = model.parameters()
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)

# Train ===============================================================================
l_cost = np.inf
t = 1
cost = np.inf
while cost > 0.02 and t <= max_epochs:
    permutation = torch.randperm(xtrain.size()[0])
    avg_loss1 = 0
    avg_loss2 = 0
    for i in range(0, N, mb_size):
        indices = permutation[i:i + mb_size]
        datax = xtrain[indices, :, :, :]
        # with autograd.detect_anomaly():
        output1, _, _ = net1(datax)
        loss1, mse1, h1, _, _, a1 = h1_loss(output1, datax)
        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer.step()
        avg_loss1 += loss1.detach().cpu().numpy()  # sum over whole dataset
    cost = avg_loss1

    # remember the lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    save_checkpoint({
        'epochs': t,
        'net1_state_dict': net1.state_dict(),
        'net3_state_dict': net3.state_dict(),
        'l_cost': l_cost,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    print(t, cost)

    t += 1
print('RKM finished Training. Lowest cost: {}'.format(l_cost))

output1 = net1(xtrain)
_, _, h1, s1, U1, a1 = h1_loss(output1, xtrain)
dir = 'kpca_trained_RKM_{}.tar'.format(time.strftime("%Y%m%d-%H%M"))

# Save Model and tensors
if not os.path.exists('out/'):
    os.makedirs('out/')

torch.save({'net1': net1,
            'net3': net3,
            'net1_state_dict': net1.state_dict(),
            'net3_state_dict': net3.state_dict(),
            'xtrain': xtrain,
            'capacity': capacity,
            'h1': h1, 'N': N, 's1': s1, 'U1': U1}, 'out/{}'.format(dir))
print('Saved Model name: {}'.format(dir))

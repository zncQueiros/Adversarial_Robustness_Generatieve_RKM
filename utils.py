import torch
import os
import time
import copy
import numpy as np
from torch.autograd import Variable

from dataloader import get_mnist_dataloader

device = torch.device("cpu")

h_dim = 10

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


def adv_train(X, y, model, criterion, adversary):
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

    X_adv = adversary.perturb(X.numpy(), y)

    return torch.from_numpy(X_adv)


def final_compute(args, net1, net2, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _ = get_mnist_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))
    ytr = net2(xt.dataset.targets[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), h, s


def truncated_normal(mean=0.0, stddev=1.0, m=1):
    '''
    The generated values follow a normal distribution with specified
    mean and standard deviation, except that values whose magnitude is
    more than 2 standard deviations from the mean are dropped and
    re-picked. Returns a vector of length m
    '''
    samples = []
    for i in range(m):
        while True:
            sample = np.random.normal(mean, stddev)
            if np.abs(sample) <= 2 * stddev:
                break
        samples.append(sample)
    assert len(samples) == m, "something wrong"
    if m == 1:
        return samples[0]
    else:
        return np.array(samples)


def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(Variable(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)

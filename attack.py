import copy
import numpy as np
# from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch.autograd import Variable


# --- White-box attacks ---

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = Variable(torch.from_numpy(X), requires_grad=True)
        y_var = Variable(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.1, k=40, a=0.01, amount=50, random_start=True, device=None):
        """
        Attack parameter initialization. The attack performs of
        size a with amount of 50, while always staying within epsilon
        from the initial point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.amount = amount
        self.rand = random_start
        self.device = device
        self.loss_fn = nn.MSELoss()

    def perturb(self, X_nat):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat = X_nat.to(self.device).float()

        if self.rand:
            X = X_nat + torch.from_numpy(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone()

        # for i in range(self.k):
        it = 1
        while torch.norm(X - X_nat).item() < self.amount and it < self.k:
            X_var = Variable(X, requires_grad=True)

            scores = self.model(X_var)
            loss = self.loss_fn(scores, X_var)
            loss.backward()
            grad = X_var.grad.data

            X = X + self.a * torch.sign(grad)

            X = torch.max(torch.min(X, X_nat + self.epsilon), X_nat - self.epsilon)
            X = torch.clamp(X, 0, 1)  # ensure valid pixel range
            it += 1
        return X


# --- Black-box attacks ---

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = Variable(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev) + ind] = X_sub[ind] + lmbda * grad_val  # ???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

from dataloader import get_mnist_dataloader
from utils import *
import numpy as np
import torch
import urllib.request
import os.path
import pyiqa
from Architecture_VAE import Encoder, Decoder, Model  # Architecture_VAE_large
from torchmetrics.image import StructuralSimilarityIndexMeasure


class AttackAdvModel:

    def __init__(self, filename, latent_dim=128, vae=False):
        self.multiview = None
        self.V = None
        self.U = None
        self.s = None
        self.net3 = None
        self.net1 = None
        self.model = None
        self.ytrain = None
        self.xtrain = None
        self.xt = None
        self.opt_gen = None
        self.h = None
        self.filename = filename
        self.latent_dim = latent_dim
        self.vae = vae

        # Parameters
        self.nbr_image = 0
        self.nbr_image_target = 1  # index of chosen target image for target attack

        self.my_lambda = 0.1  # 10 for type 1 attack, and 0.1 for type 2 and target  #trade-off between distortion
        # input/output and difference output/input (depending on attack 1 or 2)
        self.nbr_iterations = 100  # iterations to compute the perturbation
        self.eta = 0.1  # learning rate, proportion of the gradient added to the image

        self.it_attack = None  # number of iterations till limit of perturbation is attained

        # Distortion input and output
        # measured with the Frobenius norm
        self.distortion_input = []
        self.distortion_output = []

        self.lpips_input = []
        self.lpips_output = []

        self.ssim_input = []
        self.ssim_output = []

        # images
        self.image_ori = None
        self.image_adv = None
        self.image_adv_back = None
        self.image_target = None
        self.type1 = None

        self.lpips = pyiqa.create_metric('lpips', device='cpu')

        # ssim function: Structural Similarity Index Measure
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def load_model_vae(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--filename', type=str, default="out/MNIST_adv_RKM_h128_epoch300_perturb50.tar", help='Enter Filename')
        parser.add_argument('--device', type=str, default='cpu', help='Device type: cuda or cpu')
        sd_mdl = torch.load('{}'.format("out/MNIST_adv_RKM_h128_epoch300_perturb50.tar"), map_location='cpu')
        self.opt_gen = parser.parse_args()
        if 'opt' in sd_mdl:
            opt = sd_mdl['opt']
            # self.opt_gen = argparse.Namespace(**vars(opt), **vars(self.opt_gen))
            opt_dict = vars(opt)
            opt_dict.update(vars(self.opt_gen))  # Update opt_gen with values from opt, prioritizing opt_gen
            self.opt_gen = argparse.Namespace(**opt_dict)
        else:
            self.opt_gen.mb_size = 200

        self.opt_gen.shuffle = False
        self.h = sd_mdl['h'].double().detach().cpu()
        self.xt, _, _ = get_mnist_dataloader(args=self.opt_gen)  # loading data without shuffle
        self.xtrain = self.xt.dataset.train_data[:self.h.shape[0], :, :, :]  # 60.000, 1, 28, 28; h shape 0: 5000
        self.ytrain = self.xt.dataset.targets[:self.h.shape[0], :]
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(latent_dim=self.latent_dim)
        self.model = Model(Encoder=encoder, Decoder=decoder).to("cpu")
        loaded_state_dict = torch.load(self.filename, map_location=torch.device('cpu'))
        filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered_state_dict)

    def load_model_rkm(self, multiview=False):
        # Load a Pre-trained model or saved model ====================
        self.multiview = multiview
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--filename', type=str, default=self.filename, help='Enter Filename')
        parser.add_argument('--device', type=str, default='cpu', help='Device type: cuda or cpu')
        self.opt_gen = parser.parse_args()

        print(f" my filename {self.opt_gen.filename}")

        sd_mdl = torch.load('{}'.format(self.filename), map_location='cpu')

        self.net1 = sd_mdl['net1'].double().cpu()
        self.net3 = sd_mdl['net3'].double().cpu()

        self.net1.load_state_dict(sd_mdl['net1_state_dict'])
        self.net3.load_state_dict(sd_mdl['net3_state_dict'])
        self.h = sd_mdl['h'].double().detach().cpu()
        self.s = sd_mdl['s'].double().detach().cpu()
        if multiview is True:
            self.V = sd_mdl['V'].double().detach().cpu()
        self.U = sd_mdl['U'].double().detach().cpu()

        if 'opt' in sd_mdl:
            opt = sd_mdl['opt']
            opt_dict = vars(opt)
            opt_dict.update(vars(self.opt_gen))  # Update opt_gen with values from opt, prioritizing opt_gen
            self.opt_gen = argparse.Namespace(**opt_dict)
        else:
            self.opt_gen.mb_size = 200

        self.opt_gen.shuffle = False
        self.xt, _, _ = get_mnist_dataloader(args=self.opt_gen)  # loading data without shuffle
        self.xtrain = self.xt.dataset.train_data[:self.h.shape[0], :, :, :]  # 60.000, 1, 28, 28; h shape 0: 5000
        self.ytrain = self.xt.dataset.targets[:self.h.shape[0], :]

    def __distortion_images(self, first_image, second_image):
        """
        Measure the distortion between two images using the Frobenius norm
        """

        return float(torch.linalg.vector_norm(first_image - second_image).detach().numpy())

    def __get_lpips(self, first_image, second_image):

        target_size = (256, 256)  # Adjust the target size as needed

        img1_resized = F.interpolate(first_image, size=target_size, mode='bilinear', align_corners=False)
        img2_resized = F.interpolate(second_image, size=target_size, mode='bilinear', align_corners=False)

        lpips_distortion = self.lpips(img1_resized.float(), img2_resized.float()).item()

        return lpips_distortion

    def __get_ssim(self, target_image, pred_image):
        return self.ssim(pred_image, target_image).item()

    def __generate_back(self, x):
        """
        Receives an imput as image, goes to the feature space, then latent space, afteerwards back to the feature space and back to the image space.
        Travels the complete model to generate back the image.
        Input: (1, 1, 28, 28) (first dimension needed for the neural network)
        Output: (1, 28, 28)
        """
        if self.vae:
            # with torch.no_grad():
            datax_gen = self.model(x.float())[0]
        else:
            # go to the feature space
            output1 = self.net1(x.double())

            # compute (Lambda-V_T*V)^-1
            if self.multiview:
                VT_V = torch.mm(self.V.T, self.V)
                Lambda = torch.diag(self.s[:self.h.size(1)])
                lam_VTV = torch.linalg.inv(Lambda - VT_V)
                latent1 = torch.mm(torch.mm(output1, self.U), lam_VTV.T)
            else:
                Lambda = torch.diag(self.s[:self.h.size(1)])
                lam_inv = torch.linalg.inv(Lambda)
                latent1 = torch.mm(torch.mm(output1, self.U), lam_inv.T)
            # compute h = (Lambda-V_T*V)^-1*U_T*net1(x)
            # note here the computation is done with everything transposed

            # go back to the feature space U*h
            my_input_net3 = torch.mm(latent1, torch.t(self.U))

            # go back to the image space
            datax_gen = self.net3(my_input_net3[0, :])

        return datax_gen

    # Algorithm coming from the paper of Sun
    # define the loss function to generate the adversarial image

    def __loss_function(self, x_ori, x, lam, type1=True):
        """ 
        Loss function to generate the adversarial image where input different and output same. (type 1 attack)
        Loss function to generate the adversarial image where input similar and output different. (type 2 attack) 
        """
        self.image_adv_back = self.__generate_back(x)
        my_loss = torch.linalg.vector_norm(self.image_adv_back - x_ori) - lam * torch.linalg.vector_norm(
            x - x_ori)  # torch.linalg.vector_norm

        if (not type1):
            my_loss = -1 * my_loss  # change sign of the loss function in case type 2 attack

        return my_loss

    def __loss_function_PGD(self, x_ori, x):
        """ 
        PGD attack, Madry et al. (extension of the attack of Goodfellow Fast gradient method) 
        """
        x_back = self.__generate_back(x)

        my_loss = -torch.linalg.vector_norm(x_back - x_ori)  # torch.linalg.vector_norm

        return my_loss

    def __loss_function_target(self, x_ori, x_target, x, lam):
        """ Loss function to generate the adversarial image where input similar and output different and equal to a target image. (type 2 attack) """

        self.image_adv_back = self.__generate_back(x)
        my_loss = torch.linalg.vector_norm(self.image_adv_back - x_target) + lam * torch.linalg.vector_norm(
            x - x_ori)  # torch.linalg.vector_norm

        return my_loss

    def __get_latent(self, x):

        # go to the feature space
        output1 = self.net1(x)

        # compute (Lambda-V_T*V)^-1
        VT_V = torch.mm(self.V.T, self.V)
        Lambda = torch.diag(self.s[:self.h.size(1)])
        lam_VTV = torch.linalg.inv(Lambda - VT_V)

        # compute h = (Lambda-V_T*V)^-1*U_T*net1(x)
        # note here the computation is done with everything transposed
        latent1 = torch.mm(torch.mm(output1, self.U), lam_VTV.T)

        return latent1

    def __loss_function_Tabacof(self, x_ori, x_target, x, lam):
        """ Loss function to generate the adversarial image where latent code is similar to the target latent code and perturbation as small as possible (Tabacof) """

        h_image = self.__get_latent(x)
        h_target = self.__get_latent(x_target)

        my_loss = torch.linalg.vector_norm(h_image - h_target) + lam * torch.linalg.vector_norm(
            x - x_ori)  # torch.linalg.vector_norm

        return my_loss

    def adversarial_attack(self, nbr_image, nbr_it, lam_fun, eta_fun, type1):
        """
        Adversarial attack using algorithm from the paper of Sun (type 1 attack)

        adv_image_ori = original unperturbed image
        adv_image = image that will be perturbed
        nbr_it = number of iterations (typical value 10)
        lam_fun = lambda value = trade-off between the two terms (typical value 10 for type 1 and 0.1 for type 2)
        eta = rate to apply the perturbation on the image, "learning rate" (typical value 0.1)
        type1 = chose between type 1 and type 2 attack 
        """

        self.nbr_image = nbr_image
        self.type1 = type1

        x_ori = self.xt.dataset.train_data[self.nbr_image, :, :, :]
        y_ori = self.xt.dataset.targets[self.nbr_image, :]

        datax, _ = x_ori.to(self.opt_gen.device), y_ori.to(self.opt_gen.device)  # _ = datay
        image = datax[None, :, :, :]
        assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

        # starting image for the adversarial attack (equal to the original image)
        self.image_ori = image.clone().detach()

        # initialize the adversarial image that will be perturbed
        self.image_adv = torch.tensor(self.image_ori, requires_grad=True)

        # initialize the distortion lists
        self.distortion_input = []
        self.distortion_output = []
        self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_ori))
        self.distortion_output.append(self.__distortion_images(self.image_ori, self.__generate_back(self.image_ori)))

        self.lpips_input = []
        self.lpips_output = []
        self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_ori))
        self.lpips_output.append(self.__get_lpips(self.image_ori, self.__generate_back(self.image_ori)))

        self.ssim_input = []
        self.ssim_output = []
        self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_ori))
        self.ssim_output.append(self.__get_ssim(self.image_ori, self.__generate_back(self.image_ori)))

        # attack on the image
        for i in range(nbr_it):
            self.image_adv = self.image_adv.clone().detach().requires_grad_(True)
            loss_value = self.__loss_function(self.image_ori, self.image_adv, lam_fun, type1)
            loss_value.backward()  # 1.0
            self.image_adv = self.image_adv - eta_fun * self.image_adv.grad
            torch.clamp(self.image_adv, 0, 1)  # clamp all values between 0 and 1

            if (i + 1) % 1 == 0:
                self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_adv))
                self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_adv))

                self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_adv))
                self.lpips_output.append(self.__get_lpips(self.image_ori, self.image_adv_back))

                self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_adv))
                self.ssim_output.append(self.__get_ssim(self.image_ori, self.image_adv_back))

        return "Attack succeeded"

    def adversarial_attack_lim(self, nbr_image, max_input_perturbation, lam_fun, eta_fun, type1):
        """
        Adversarial attack using algorithm from the paper of Sun (type 1 attack)

        with lim = limited in the amount of input perturbation instead of the number of iterations

        adv_image_ori = original unperturbed image
        adv_image = image that will be perturbed
        nbr_it = number of iterations (typical value 10)
        lam_fun = lambda value = trade-off between the two terms (typical value 10 for type 1 and 0.1 for type 2)
        eta = rate to apply the perturbation on the image, "learning rate" (typical value 0.1)
        type1 = chose between type 1 and type 2 attack 
        """

        self.nbr_image = nbr_image
        self.type1 = type1

        x_ori = self.xt.dataset.train_data[self.nbr_image, :, :, :]
        y_ori = self.xt.dataset.targets[self.nbr_image, :]

        datax, _ = x_ori.to(self.opt_gen.device), y_ori.to(self.opt_gen.device)  # _ = datay
        image = datax[None, :, :, :]
        assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

        # starting image for the adversarial attack (equal to the original image)
        self.image_ori = image.clone().detach()

        # initialize the adversarial image that will be perturbed
        self.image_adv = torch.tensor(self.image_ori, requires_grad=True)

        # initialize the distortion lists
        self.distortion_input = []
        self.distortion_output = []
        self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_ori))
        self.distortion_output.append(self.__distortion_images(self.image_ori, self.__generate_back(self.image_ori)))

        self.lpips_input = []
        self.lpips_output = []
        self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_ori))
        self.lpips_output.append(self.__get_lpips(self.image_ori, self.__generate_back(self.image_ori)))

        self.ssim_input = []
        self.ssim_output = []
        self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_ori))
        self.ssim_output.append(self.__get_ssim(self.image_ori, self.__generate_back(self.image_ori)))

        nbr_it = 0

        # attack on the image
        while self.distortion_input[nbr_it] < max_input_perturbation and nbr_it < 100:
            self.image_adv = self.image_adv.clone().detach().requires_grad_(True)
            loss_value = self.__loss_function(self.image_ori, self.image_adv, lam_fun, type1)
            loss_value.backward()  # 1.0
            self.image_adv = self.image_adv - eta_fun * self.image_adv.grad
            torch.clamp(self.image_adv, 0, 1)  # clamp all values between 0 and 1

            if (nbr_it + 1) % 1 == 0:
                self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_adv))
                # self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_adv))
                self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_ori))

                self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_adv))
                # self.lpips_output.append(self.__get_lpips(self.image_adv_back, self.image_adv))
                self.lpips_output.append(self.__get_lpips(self.image_adv_back, self.image_ori))

                self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_adv))
                self.ssim_output.append(self.__get_ssim(self.image_ori, self.image_adv_back))

            nbr_it += 1

        self.it_attack = nbr_it
        return "Attack succeeded"

    def adversarial_attack_target(self, nbr_image, nbr_image_target, nbr_it, lam_fun, eta_fun):
        """
        Adversarial attack using algorithm from the paper of Sun (type 1 attack)

        adv_image_ori = original unperturbed image
        adv_image = image that will be perturbed
        x_target = target image to which we want that the output of the adversarial image is equal
        nbr_it = number of iterations (typical value 10)
        lam_fun = lambda value = trade-off between the two terms (typical value 0.1)
        eta = rate to apply the perturbation on the image, "learning rate" (typical value 0.1)
        """

        self.type1 = False

        self.nbr_image = nbr_image
        self.nbr_image_target = nbr_image_target

        x_ori = self.xt.dataset.train_data[self.nbr_image, :, :, :]
        y_ori = self.xt.dataset.targets[self.nbr_image, :]

        datax, _ = x_ori.to(self.opt_gen.device), y_ori.to(self.opt_gen.device)  # _ = datay
        image = datax[None, :, :, :]
        assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

        # starting image for the adversarial attack (equal to the original image)
        self.image_ori = image.clone().detach()

        # create target image
        xchosen_target = self.xt.dataset.train_data[self.nbr_image_target, :, :, :]
        self.image_target = xchosen_target.to(self.opt_gen.device)

        self.image_target = self.image_target[None, :, :, :]
        assert self.image_target.shape == (1, 1, 28, 28)

        # initialize the adversarial image that will be perturbed
        self.image_adv = torch.tensor(self.image_ori, requires_grad=True)

        # initialize the Distortion lists
        self.distortion_input = []
        self.distortion_output = []
        self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_ori))
        self.distortion_output.append(self.__distortion_images(self.image_ori, self.__generate_back(self.image_ori)))

        self.lpips_input = []
        self.lpips_output = []
        self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_ori))
        self.lpips_output.append(self.__get_lpips(self.image_ori, self.__generate_back(self.image_ori)))

        self.ssim_input = []
        self.ssim_output = []
        self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_ori))
        self.ssim_output.append(self.__get_ssim(self.image_ori, self.__generate_back(self.image_ori)))

        # attack on the image
        for i in range(nbr_it):
            self.image_adv = self.image_adv.clone().detach().requires_grad_(True)
            loss_value = self.__loss_function_target(self.image_ori, self.image_target, self.image_adv, lam_fun)
            loss_value.backward()  # 1.0
            self.image_adv = self.image_adv - eta_fun * self.image_adv.grad
            torch.clamp(self.image_adv, 0, 1)  # clamp all values between 0 and 1

            if ((i + 1) % 1 == 0):
                self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_adv))
                self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_ori))

                self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_adv))
                self.lpips_output.append(self.__get_lpips(self.image_adv_back, self.image_ori))

                self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_adv))
                self.ssim_output.append(self.__get_ssim(self.image_ori, self.image_adv_back))

            nbr_it += 1

        self.it_attack = nbr_it

        return "Attack succeeded"

    def adversarial_attack_target_lim(self, nbr_image, nbr_image_target, max_input_perturbation, lam_fun, eta_fun):
        """
        Adversarial attack using algorithm from the paper of Sun (type 1 attack)

        adv_image_ori = original unperturbed image
        adv_image = image that will be perturbed
        x_target = target image to which we want that the output of the adversarial image is equal
        nbr_it = number of iterations (typical value 10)
        lam_fun = lambda value = trade-off between the two terms (typical value 0.1)
        eta = rate to apply the perturbation on the image, "learning rate" (typical value 0.1)
        """

        self.type1 = False

        self.nbr_image = nbr_image
        self.nbr_image_target = nbr_image_target

        x_ori = self.xt.dataset.train_data[self.nbr_image, :, :, :]
        y_ori = self.xt.dataset.targets[self.nbr_image, :]

        datax, _ = x_ori.to(self.opt_gen.device), y_ori.to(self.opt_gen.device)  # _ = datay
        image = datax[None, :, :, :]
        assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

        # starting image for the adversarial attack (equal to the original image)
        self.image_ori = image.clone().detach()

        # create target image
        xchosen_target = self.xt.dataset.train_data[self.nbr_image_target, :, :, :]
        self.image_target = xchosen_target.to(self.opt_gen.device)

        self.image_target = self.image_target[None, :, :, :]
        assert self.image_target.shape == (1, 1, 28, 28)

        # initialize the adversarial image that will be perturbed
        self.image_adv = torch.tensor(self.image_ori, requires_grad=True)

        # initialize the Distortion lists
        self.distortion_input = []
        self.distortion_output = []
        self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_ori))
        self.distortion_output.append(self.__distortion_images(self.image_ori, self.__generate_back(self.image_ori)))

        self.lpips_input = []
        self.lpips_output = []
        self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_ori))
        self.lpips_output.append(self.__get_lpips(self.image_ori, self.__generate_back(self.image_ori)))

        self.ssim_input = []
        self.ssim_output = []
        self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_ori))
        self.ssim_output.append(self.__get_ssim(self.image_ori, self.__generate_back(self.image_ori)))

        nbr_it = 0

        # attack on the image
        while self.distortion_input[nbr_it] < max_input_perturbation and nbr_it < 100:
            self.image_adv = self.image_adv.clone().detach().requires_grad_(True)
            loss_value = self.__loss_function_target(self.image_ori, self.image_target, self.image_adv, lam_fun)
            loss_value.backward()  # 1.0
            self.image_adv = self.image_adv - eta_fun * self.image_adv.grad
            torch.clamp(self.image_adv, 0, 1)  # clamp all values between 0 and 1

            if (nbr_it + 1) % 1 == 0:
                self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_adv))
                self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_ori))

                self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_adv))
                self.lpips_output.append(self.__get_lpips(self.image_adv_back, self.image_ori))

                self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_adv))
                self.ssim_output.append(self.__get_ssim(self.image_ori, self.image_adv_back))

            nbr_it += 1

        self.it_attack = nbr_it

        return "Attack succeeded"

    def adversarial_attack_Tabacof(self, nbr_image, nbr_image_target, nbr_it, lam_fun, eta_fun):
        """
        Adversarial attack using algorithm from the paper of Tabacof

        adv_image_ori = original unperturbed image
        adv_image = image that will be perturbed
        nbr_it = number of iterations (typical value 900)
        lam_fun = lambda value = trade-off between the two terms (typical value 0.01)
        eta = rate to apply the perturbation on the image, "learning rate" (typical value 0.1)
        type1 = chose between type 1 and type 2 attack
        """

        self.type1 = False

        self.nbr_image = nbr_image
        self.nbr_image_target = nbr_image_target

        x_ori = self.xt.dataset.train_data[self.nbr_image, :, :, :]
        y_ori = self.xt.dataset.targets[self.nbr_image, :]

        datax, _ = x_ori.to(self.opt_gen.device), y_ori.to(self.opt_gen.device)  # _ = datay
        image = datax[None, :, :, :]
        assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

        # starting image for the adversarial attack (equal to the original image)
        self.image_ori = image.clone().detach()

        # create target image
        xchosen_target = self.xt.dataset.train_data[self.nbr_image_target, :, :, :]
        self.image_target = xchosen_target.to(self.opt_gen.device)

        self.image_target = self.image_target[None, :, :, :]
        assert self.image_target.shape == (1, 1, 28, 28)

        # initialize the adversarial image that will be perturbed
        self.image_adv = torch.tensor(self.image_ori, requires_grad=True)

        # initialize the Distortion lists
        self.distortion_input = []
        self.distortion_output = []
        self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_ori))
        self.distortion_output.append(self.__distortion_images(self.image_ori, self.__generate_back(self.image_ori)))

        self.lpips_input = []
        self.lpips_output = []
        self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_ori))
        self.lpips_output.append(self.__get_lpips(self.image_ori, self.__generate_back(self.image_ori)))

        self.ssim_input = []
        self.ssim_output = []
        self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_ori))
        self.ssim_output.append(self.__get_ssim(self.image_ori, self.__generate_back(self.image_ori)))

        # attack on the image
        for i in range(nbr_it):
            self.image_adv = self.image_adv.clone().detach().requires_grad_(True)
            loss_value = self.__loss_function_Tabacof(self.image_ori, self.image_target, self.image_adv, lam_fun)
            loss_value.backward()  # 1.0
            self.image_adv = self.image_adv - eta_fun * self.image_adv.grad
            torch.clamp(self.image_adv, 0, 1)  # clamp all values between 0 and 1

            if ((i + 1) % 1 == 0):
                self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_adv))
                self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_adv))

                self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_adv))
                self.lpips_output.append(self.__get_lpips(self.image_adv_back, self.image_adv))

                self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_adv))
                self.ssim_output.append(self.__get_ssim(self.image_ori, self.image_adv_back))

        return "attack succeeded"

    def adversarial_attack_PGD(self, nbr_image, nbr_it, alpha):
        """
        PGD attack, Madry et al. (extension of the attack of Goodfellow Fast gradient method) 
        Maximize the difference between the original image and the reconstruction of the adversarial sample
        Add a small perturbation in every direction with the sign function

        adv_image_ori = original unperturbed image
        adv_image = image that will be perturbed
        nbr_it = number of iterations (typical value 15)
        alpha = rate to apply the perturbation on the image, "learning rate" (typical value=0.01)
        type1 = chose between type 1 and type 2 attack
        """

        self.nbr_image = nbr_image
        self.type1 = False

        x_ori = self.xt.dataset.train_data[self.nbr_image, :, :, :]
        y_ori = self.xt.dataset.targets[self.nbr_image, :]

        datax, _ = x_ori.to(self.opt_gen.device), y_ori.to(self.opt_gen.device)  # _ = datay
        image = datax[None, :, :, :]
        assert image.shape == (1, 1, 28, 28)  # fourth dimension to be used in the neural network

        # starting image for the adversarial attack (equal to the original image)
        self.image_ori = image.clone().detach()

        # initialize the adversarial image that will be perturbed
        self.image_adv = torch.tensor(self.image_ori, requires_grad=True)

        # initialize the distortion lists
        self.distortion_input = []
        self.distortion_output = []
        self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_ori))
        self.distortion_output.append(self.__distortion_images(self.image_ori, self.__generate_back(self.image_ori)))

        self.lpips_input = []
        self.lpips_output = []
        self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_ori))
        self.lpips_output.append(self.__get_lpips(self.image_ori, self.__generate_back(self.image_ori)))

        self.ssim_input = []
        self.ssim_output = []
        self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_ori))
        self.ssim_output.append(self.__get_ssim(self.image_ori, self.__generate_back(self.image_ori)))

        # attack on the image
        for i in range(nbr_it):
            self.image_adv = self.image_adv.clone().detach().requires_grad_(True)
            loss_value = self.__loss_function_PGD(self.image_ori, self.image_adv)
            loss_value.backward()  # 1.0
            self.image_adv = self.image_adv - alpha * torch.sign(self.image_adv.grad)
            torch.clamp(self.image_adv, 0, 1)  # clamp all values between 0 and 1

            if ((i + 1) % 1 == 0):
                self.distortion_input.append(self.__distortion_images(self.image_ori, self.image_adv))
                self.distortion_output.append(self.__distortion_images(self.image_adv_back, self.image_adv))

                self.lpips_input.append(self.__get_lpips(self.image_ori, self.image_adv))
                self.lpips_output.append(self.__get_lpips(self.image_adv_back, self.image_adv))

                self.ssim_input.append(self.__get_ssim(self.image_ori, self.image_adv))
                self.ssim_output.append(self.__get_ssim(self.image_ori, self.image_adv_back))

        return "Attack succeeded"

    def save_distortion_arrays(self):
        # Convert list to NumPy array
        distortion_input_array = np.array(self.distortion_input)
        distortion_output_array = np.array(self.distortion_output)

        np.save('lists/sim_in_targ_5002_01_10_02_rob.npy', distortion_input_array)
        np.save('lists/sim_out_targ_5002_01_10_02_rob.npy', distortion_output_array)

    def plot_ori_back(self):
        """
        Plot original image and the reconstructed image
        """
        image_ori_gen = self.__generate_back(self.image_ori)
        image_ori_gen = image_ori_gen.reshape(1, 28, 28)

        model_name = "genRKM"
        if self.vae:
            model_name = "VAE"

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.image_ori[0, 0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[1].imshow(image_ori_gen[0, :].detach().numpy(), cmap='Greys_r', vmin=0, vmax=1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.suptitle(f"Original and reconstructed image with {model_name}")
        plt.show()

    def plot_ori_back_adv_back(self):
        # ouput of the adversarial input sample/output of the perturbed image
        image_ori_gen = self.__generate_back(self.image_ori)
        image_ori_gen = image_ori_gen.reshape(1, 28, 28)

        image_adv_gen = self.image_adv_back
        image_adv_gen = image_adv_gen.clone().detach().reshape(1, 28, 28)
        image_adv_reshaped = self.image_adv.detach().reshape(1, 28, 28)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(self.image_ori[0, 0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[0, 1].imshow(image_ori_gen[0, :].detach().numpy(), cmap='Greys_r', vmin=0, vmax=1)
        ax[1, 0].imshow(image_adv_reshaped[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[1, 1].imshow(image_adv_gen[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        plt.suptitle(
            "Original image, original image reconstructed,\n adversarial image and adversarial image reconstructed")
        plt.show()

    def plot_adv_back(self):

        image_adv_gen = self.image_adv_back
        image_adv_gen = image_adv_gen.clone().detach().reshape(1, 28, 28)
        image_adv_reshaped = self.image_adv.detach().reshape(1, 28, 28)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_adv_reshaped[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[1].imshow(image_adv_gen[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.suptitle("Adversarial image and the reconstruction")
        plt.show()

    def plot_adv_diff(self):

        image_adv_reshaped = self.image_adv.detach().reshape(1, 28, 28)
        image_ori_reshape = self.image_ori.detach().reshape(1, 28, 28)

        # distortion = image_ori_reshape - image_adv_reshaped

        distortion = image_adv_reshaped - image_ori_reshape

        # Calculate the minimum and maximum values of the tensor
        # min_val = torch.min(distortion)
        # max_val = torch.max(distortion)

        # distortion = (distortion - min_val)/(max_val-min_val)

        # distortion = (distortion + 1)/2

        # distortion = torch.abs(distortion)

        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(image_adv_reshaped[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[1].imshow(distortion[0, :], cmap='bwr')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.suptitle("Adversarial image and the used distortion")
        plt.show()

    def plot_ori_adv_diff(self):

        image_adv_reshaped = self.image_adv.detach().reshape(1, 28, 28)
        image_ori_reshape = self.image_ori.detach().reshape(1, 28, 28)

        image_adv_gen = self.image_adv_back
        image_adv_gen = image_adv_gen.clone().detach().reshape(1, 28, 28)

        distortion = image_adv_reshaped - image_ori_reshape

        # Calculate the minimum and maximum values of the tensor
        # min_val = torch.min(distortion)
        # max_val = torch.max(distortion)

        # distortion = (distortion - min_val)/(max_val-min_val)

        # distortion = (distortion + 1)/2

        # distortion = torch.abs(distortion)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(self.image_ori[0, 0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[0, 1].imshow(image_adv_reshaped[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[1, 0].imshow(distortion[0, :], cmap='bwr')
        ax[1, 1].imshow(image_adv_gen[0, :], cmap='Greys_r', vmin=0, vmax=1)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        plt.suptitle("Original image and adversarial image\nPerturbation and reconstructed image")
        plt.show()

    def plot_distortion(self):

        type_of_attack = "type 1"

        if (not self.type1):
            type_of_attack = "type 2"

        # Plotting the curves with specified colors
        plt.plot(range(len(self.distortion_input)), self.distortion_input, color='red', label='Input distortion')
        plt.plot(range(len(self.distortion_output)), self.distortion_output, color='blue', label='Output distortion')

        # Adding legend
        plt.legend()

        # Adding labels and title
        plt.xlabel('Nbr of iterations')
        plt.ylabel('Distortion (Frob Norm)')
        plt.title(
            f'Adversarial attack {type_of_attack} ({len(self.distortion_input) - 1} iterations) \nDistortion in the input and output with original image')

        # Display plot
        plt.show()

    def plot_lpips(self):

        type_of_attack = "type 1"

        if not self.type1:
            type_of_attack = "type 2"

        # Plotting the curves with specified colors
        plt.plot(range(len(self.lpips_input)), self.lpips_input, color='red', label='Input distortion (lpips)')
        plt.plot(range(len(self.lpips_output)), self.lpips_output, color='blue', label='Output distortion (lpips)')

        # Adding legend
        plt.legend()

        # Adding labels and title
        plt.xlabel('Nbr of iterations')
        plt.ylabel('Distortion (lpips)')
        plt.title(
            f'Adversarial attack {type_of_attack} ({len(self.distortion_input) - 1} iterations) \nDistortion (LPIPS) in the input and output with original image')

        # Display plot
        plt.show()

    def get_distortion_arrays(self):

        distortion_input_array = np.array(self.distortion_input)
        distortion_output_array = np.array(self.distortion_output)

        return distortion_input_array, distortion_output_array

    def get_lpips_arrays(self):

        lpips_input_array = np.array(self.lpips_input)
        lpips_output_array = np.array(self.lpips_output)

        return lpips_input_array, lpips_output_array

    def get_ssim_arrays(self):

        ssim_input_array = np.array(self.ssim_input)
        ssim_output_array = np.array(self.ssim_output)

        return ssim_input_array, ssim_output_array

    def get_final_lpips(self):

        target_size = (256, 256)  # Adjust the target size as needed

        image_ori_gen = self.__generate_back(self.image_ori)
        image_adv_gen = self.image_adv_back  # self.__generate_back(self.image_adv)

        img_ori_resized = F.interpolate(self.image_ori, size=target_size, mode='bilinear', align_corners=False)
        img_ori_gen_resized = F.interpolate(image_ori_gen, size=target_size, mode='bilinear', align_corners=False)
        img_adv_resized = F.interpolate(self.image_adv, size=target_size, mode='bilinear', align_corners=False)
        img_adv_gen_resized = F.interpolate(image_adv_gen, size=target_size, mode='bilinear', align_corners=False)

        """
        print("LPIPS (orig_input - adv_input): ", lpips(img_ori_resized.float(), img_adv_resized.float()).item())
        print("LPIPS (orig_input - orig_output): ", lpips(img_ori_resized.float(), img_ori_gen_resized.float()).item())
        print("LPIPS (adv_input - adv_output): ", lpips(img_adv_resized.float(), img_adv_gen_resized.float()).item())
        print("LPIPS (orig_output - adv_output): ", lpips(img_ori_gen_resized.float(), img_adv_gen_resized.float()).item())
        """

        ori_adv = self.lpips(img_ori_resized.float(), img_adv_resized.float()).item()
        ori_ori_gen = self.lpips(img_ori_resized.float(), img_ori_gen_resized.float()).item()
        adv_adv_gen = self.lpips(img_adv_resized.float(), img_adv_gen_resized.float()).item()
        ori_gen_adv_gen = self.lpips(img_ori_gen_resized.float(), img_adv_gen_resized.float()).item()

        return ori_adv, ori_ori_gen, adv_adv_gen, ori_gen_adv_gen

    def get_it_attack(self):
        return self.it_attack




import math

import matplotlib.pyplot as plt

from class_Adv_attack_genRKM import AttackAdvModel
from utils import *

filenames_adv = {
    "filename_adv_vae": "out/VAE_0001LR_200epo_5kN_100BS_CNN_adv.pth",
    "filename_adv_rkm": "out/MNIST_adv_RKM_full_h128_epoch300_perturb50.tar",
    "filename_pkpca_adv_rkm": "out/MNIST_adv_RKM_pkpca.tar",
    "filename_lat_rkm": "out/MNIST_LAT_RKM.tar",
    "filename_pertview": "out/MNIST_adv_RKM_perturbation_view.tar",
    "filename_ds_lat_rkm": "out/MNIST_ds_lat_RKM.tar",
}

filenames_vanilla = {
    "filename_vae": "out/VAE_0001LR_200epo_5kN_100BS_CNN.pth",
    "filename_rkm": "out/MNIST_vanilla_rkm.tar",
    "filename_pkpca_rkm": "out/MNIST_RKM_pkpca.tar",
    "filename_ds_rkm": "out/MNIST_ds_RKM.tar",
}


class ModelComparison:

    def __init__(self, filenames, adv=True):
        self.adv = adv
        if self.adv:
            self.filename_adv_vae = filenames.get("filename_adv_vae")
            self.filename_adv_rkm = filenames.get("filename_adv_rkm")
            self.filename_pkpca_adv_rkm = filenames.get("filename_pkpca_adv_rkm")
            self.filename_lat_rkm = filenames.get("filename_lat_rkm")
            self.filename_pertview = filenames.get("filename_pertview")
            self.filename_ds_lat_rkm = filenames.get("filename_ds_lat_rkm")
        else:
            self.filename_vae = filenames.get("filename_vae")
            self.filename_rkm = filenames.get("filename_rkm")
            self.filename_pkpca_rkm = filenames.get("filename_pkpca_rkm")
            self.filename_ds_rkm = filenames.get("filename_ds_rkm")

        if self.adv:
            self.adv_vae = AttackAdvModel(self.filename_adv_vae, vae=True)
            self.adv_vae.load_model_vae()
            self.adv_rkm = AttackAdvModel(self.filename_adv_rkm)
            self.adv_rkm.load_model_rkm()
            self.pkpca_adv_rkm = AttackAdvModel(self.filename_pkpca_adv_rkm)
            self.pkpca_adv_rkm.load_model_rkm()
            self.lat_rkm = AttackAdvModel(self.filename_lat_rkm)
            self.lat_rkm.load_model_rkm()
            self.pertview = AttackAdvModel(self.filename_pertview)
            self.pertview.load_model_rkm(multiview=True)
            self.ds_lat_rkm = AttackAdvModel(self.filename_ds_lat_rkm)
            self.ds_lat_rkm.load_model_rkm()
            self.models = {
                "adv_vae": self.adv_vae,
                "adv_rkm": self.adv_rkm,
                "pkpca_adv_rkm": self.pkpca_adv_rkm,
                "lat_rkm": self.lat_rkm,
                "pertview": self.pertview,
                "ds_lat_rkm": self.ds_lat_rkm
            }

        else:
            self.vae = AttackAdvModel(self.filename_vae, vae=True)
            self.vae.load_model_vae()
            self.rkm = AttackAdvModel(self.filename_rkm)
            self.rkm.load_model_rkm()
            self.pkpca_rkm = AttackAdvModel(self.filename_pkpca_rkm)
            self.pkpca_rkm.load_model_rkm()
            self.ds_rkm = AttackAdvModel(self.filename_ds_rkm)
            self.ds_rkm.load_model_rkm()
            self.models = {
                "vae": self.vae,
                "rkm": self.rkm,
                "pkpca_rkm": self.pkpca_rkm,
                "ds_rkm": self.ds_rkm
            }

    def assess_initial_performance(self, plot=False):
        nbr_images = 25
        idx = 0
        LPIPS_init = [[] for _ in range(len(self.models))]
        distortion_init = [[] for _ in range(len(self.models))]
        SSIM_init = [[] for _ in range(len(self.models))]
        for model_name, model in self.models.items():

            for image in range(nbr_images):
                model.adversarial_attack(image, 0, 0.5, 0.1, False)

                lpips_input, lpips_output = model.get_lpips_arrays()
                distortion_input, distortion_output = model.get_distortion_arrays()
                ssim_input, ssim_output = model.get_ssim_arrays()

                LPIPS_init[idx].append(lpips_output)
                distortion_init[idx].append(distortion_output)
                SSIM_init[idx].append(ssim_output)

                if plot is True:
                    model.plot_ori_back()

            print(
                f"mean {model_name} LPIPS {np.round(np.mean(LPIPS_init[idx]), 4)} ({np.round(np.std(LPIPS_init[idx]), 4)})")
            print(
                f"mean {model_name} Distortion {np.round(np.mean(distortion_init[idx]), 4)} ({np.round(np.std(distortion_init[idx]), 4)})")
            print(
                f"mean {model_name} SSIM {np.round(np.mean(SSIM_init[idx]), 4)} ({np.round(np.std(SSIM_init[idx]), 4)})")

        idx += 1

    def assess_untarget_performance(self, test=False, plot=True):
        n_images = 25
        idx = 0
        first_image = 0
        LPIPS_untarget_input = [[] for _ in range(len(self.models))]
        distortion_untarget_input = [[] for _ in range(len(self.models))]
        SSIM_untarget_input = [[] for _ in range(len(self.models))]
        LPIPS_untarget_output = [[] for _ in range(len(self.models))]
        distortion_untarget_output = [[] for _ in range(len(self.models))]
        SSIM_untarget_output = [[] for _ in range(len(self.models))]
        if test is True:
            first_image = 5001

        for model_name, model in self.models.items():
            it = []
            for image in range(first_image, n_images):
                model.adversarial_attack_lim(image, 3, 0.1, 0.1, False)

                lpips_input, lpips_output = model.get_lpips_arrays()
                distortion_input, distortion_output = model.get_distortion_arrays()
                ssim_input, ssim_output = model.get_ssim_arrays()

                SSIM_untarget_input[idx].append(ssim_input)
                SSIM_untarget_output[idx].append(ssim_output)

                distortion_untarget_input[idx].append(distortion_input)
                distortion_untarget_output[idx].append(distortion_output)

                LPIPS_untarget_input[idx].append(lpips_input)
                LPIPS_untarget_output[idx].append(lpips_output)

                it.append(model.get_it_attack())

                # print(f"iteration {image} done")

            print(f"{model_name} iterations {np.mean(it)} ($\pm$ {np.round(np.std(it), 2)})")
            print("LPIPS        ")
            print(
                "{} ($\pm$ {})".format(np.round(average_and_std_dev_last_element(LPIPS_untarget_output[idx], True), 4),
                                       np.round(average_and_std_dev_last_element(LPIPS_untarget_output[idx], False),
                                                4)))
            print("Distortion")
            print("{} ($\pm$ {})".format(
                np.round(average_and_std_dev_last_element(distortion_untarget_output[idx], True), 3),
                np.round(average_and_std_dev_last_element(distortion_untarget_output[idx], False), 3)))
            print("SSIM")
            print("{} ($\pm$ {})".format(np.round(average_and_std_dev_last_element(SSIM_untarget_output[idx], True), 4),
                                         np.round(average_and_std_dev_last_element(SSIM_untarget_output[idx], False),
                                                  4)))
            idx += 1

        if plot is True:
            # plot distortion
            distortion_avg = [[] for _ in range(len(self.models))]
            distortion_std_dev = [[] for _ in range(len(self.models))]
            distortion_avg[1] = [1.2418315392719617, 1.6244793566921487, 2.1865046410127764, 2.8484267636919394, 3.484467379042452, 4.174801883346629, 4.8542911272130445]
            reference_points = [0, 0.5, 1, 1.5, 2, 2.5, 3]
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "adv_rkm":
                    plt.plot(reference_points, distortion_avg[i], label=model_name)
                    continue
                for element in reference_points:
                    avg, std = average_and_std(element, distortion_untarget_input[i], distortion_untarget_output[i])

                    distortion_avg[i].append(avg)
                    distortion_std_dev[i].append(std)
                if model_name == "adv_rkm":
                    print(distortion_avg[i])
                plt.plot(reference_points, distortion_avg[i], label=model_name)

            plt.xlabel('Input distortion (Frob norm)')
            plt.ylabel('Output distortion (Frob norm)')
            plt.title('Output distortion vs input distortion for compared models')
            plt.legend()

            plt.show()

            # Plot LPIPS
            lpips_avg = [[] for _ in range(len(self.models))]
            lpips_std_dev = [[] for _ in range(len(self.models))]
            lpips_avg[1] = [0.030898200944066048, 0.04082067117094994, 0.055605230927467345, 0.07615944415330887, 0.09859229385852813, 0.12921207904815674, 0.16481452107429503]
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "adv_rkm":
                    plt.plot(reference_points, lpips_avg[i], label=model_name)
                    continue
                for element in reference_points:
                    avg, std = average_and_std(element, distortion_untarget_input[i], LPIPS_untarget_output[i])

                    lpips_avg[i].append(avg)
                    lpips_std_dev[i].append(std)
                if model_name == "adv_rkm":
                    print(lpips_avg[i])
                plt.plot(reference_points, lpips_avg[i], label=model_name)

            plt.xlabel('Input distortion (Frob norm)')
            plt.ylabel('Output dissimilarity (lpips)')
            plt.title('Output dissimilarity vs input distortion for compared models')
            plt.legend()

            plt.show()

            # Plot SSIM
            ssim_avg = [[] for _ in range(len(self.models))]
            ssim_std_dev = [[] for _ in range(len(self.models))]
            ssim_avg[1] = [0.8718349719047547, 0.8576587915420533, 0.8353688883781433, 0.797534601688385, 0.7487315392494202, 0.6864834690093994, 0.6182600486278534]
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "adv_rkm":
                    plt.plot(reference_points, ssim_avg[i], label=model_name)
                    continue
                for element in reference_points:
                    avg, std = average_and_std(element, distortion_untarget_input[i], SSIM_untarget_output[i])

                    ssim_avg[i].append(avg)
                    ssim_std_dev[i].append(std)
                if model_name == "adv_rkm":
                    print(ssim_avg[i])
                plt.plot(reference_points, ssim_avg[i], label=model_name)

            plt.xlabel('Input distortion (Frob norm)')
            plt.ylabel('Output dissimilarity (ssim)')
            plt.title('Output dissimilarity vs input distortion for compared models')
            plt.legend()

            plt.show()

    def assess_target_performance(self, test=False, plot=True):
        n_images = 25
        idx = 0
        first_image = 0
        LPIPS_target_input = [[] for _ in range(len(self.models))]
        distortion_target_input = [[] for _ in range(len(self.models))]
        SSIM_target_input = [[] for _ in range(len(self.models))]
        LPIPS_target_output = [[] for _ in range(len(self.models))]
        distortion_target_output = [[] for _ in range(len(self.models))]
        SSIM_target_output = [[] for _ in range(len(self.models))]
        if test is True:
            first_image = 5001

        for model_name, model in self.models.items():
            it = []
            for image in range(first_image, n_images):
                model.adversarial_attack_target_lim(image, image + 1, 2.8, 0.1, 0.5)

                lpips_input, lpips_output = model.get_lpips_arrays()
                distortion_input, distortion_output = model.get_distortion_arrays()
                ssim_input, ssim_output = model.get_ssim_arrays()

                SSIM_target_input[idx].append(ssim_input)
                SSIM_target_output[idx].append(ssim_output)

                distortion_target_input[idx].append(distortion_input)
                distortion_target_output[idx].append(distortion_output)

                LPIPS_target_input[idx].append(lpips_input)
                LPIPS_target_output[idx].append(lpips_output)

                it.append(model.get_it_attack())

                # print(f"iteration {image} done")

            print(f"{model_name} iterations {np.mean(it)} and std dev {np.round(np.std(it), 2)}")
            print("LPIPS Distortion SSIM")
            print(np.round(average_and_std_dev_last_element(LPIPS_target_output[idx], True), 4),
                  np.round(average_and_std_dev_last_element(distortion_target_output[idx], True), 3),
                  np.round(average_and_std_dev_last_element(SSIM_target_output[idx], True), 4))
            print(np.round(average_and_std_dev_last_element(LPIPS_target_output[idx], False), 4),
                  np.round(average_and_std_dev_last_element(distortion_target_output[idx], False), 3),
                  np.round(average_and_std_dev_last_element(SSIM_target_output[idx], False), 4))

            idx += 1

        if plot is True:
            # plot distortion
            distortion_avg = [[] for _ in range(len(self.models))]
            # distortion_avg[1] = [1.2418315392719617, 1.2746887309834956, 1.5437919792482473, 2.034763020426479, 2.650346176459121, 3.285637313303732, 3.838073567494336]
            distortion_std_dev = [[] for _ in range(len(self.models))]
            reference_points = [0, 0.5, 1, 1.5, 2, 2.5, 3]
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "adv_rkm":
                    plt.plot(reference_points, distortion_avg[i], label=model_name)
                    continue
                for element in reference_points:
                    avg, std = average_and_std(element, distortion_target_input[i], distortion_target_output[i])

                    distortion_avg[i].append(avg)
                    distortion_std_dev[i].append(std)
                if model_name == "adv_rkm":
                    print(distortion_avg[i])
                plt.plot(reference_points, distortion_avg[i], label=model_name)

            plt.xlabel('Input distortion (Frob norm)')
            plt.ylabel('Output distortion (Frob norm)')
            plt.title('Output distortion vs input distortion for compared models')
            plt.legend()

            plt.show()

            # Plot LPIPS
            lpips_avg = [[] for _ in range(len(self.models))]
            lpips_std_dev = [[] for _ in range(len(self.models))]
            # lpips_avg[1] = [0.030898200944066048, 0.031635457277297975, 0.03968974657356739, 0.058335902243852614, 0.08683442085981369, 0.12232836663722992, 0.15387212842702866]
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "adv_rkm":
                    plt.plot(reference_points, lpips_avg[i], label=model_name)
                    continue
                for element in reference_points:
                    avg, std = average_and_std(element, distortion_target_input[i], LPIPS_target_output[i])

                    lpips_avg[i].append(avg)
                    lpips_std_dev[i].append(std)
                if model_name == "adv_rkm":
                    print(lpips_avg[i])
                plt.plot(reference_points, lpips_avg[i], label=model_name)

            plt.xlabel('Input distortion (Frob norm)')
            plt.ylabel('Output dissimilarity (lpips)')
            plt.title('Output dissimilarity vs input distortion for compared models')
            plt.legend()

            plt.show()

            # Plot SSIM
            ssim_avg = [[] for _ in range(len(self.models))]
            ssim_std_dev = [[] for _ in range(len(self.models))]
            ssim_avg[1] = [0.8718349719047547, 0.8614107394218444, 0.8330725049972534, 0.7932072329521179,
                           0.7406216502189636, 0.6814149618148804, 0.6267898011207581]
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "adv_rkm":
                    plt.plot(reference_points, ssim_avg[i], label=model_name)
                    continue
                for element in reference_points:
                    avg, std = average_and_std(element, distortion_target_input[i], SSIM_target_output[i])

                    ssim_avg[i].append(avg)
                    ssim_std_dev[i].append(std)
                if model_name == "adv_rkm":
                    print(ssim_avg[i])
                plt.plot(reference_points, ssim_avg[i], label=model_name)

            plt.xlabel('Input distortion (Frob norm)')
            plt.ylabel('Output dissimilarity (ssim)')
            plt.title('Output dissimilarity vs input distortion for compared models')
            plt.legend()

            plt.show()


def average_first_element(list_of_lists):
    total = 0
    count = 0
    for sublist in list_of_lists:
        if len(sublist) > 0:  # Ensure sublist is not empty
            total += sublist[0]
            count += 1
    if count == 0:
        return 0  # Return 0 if there are no valid sublists
    return total / count


# %%
def average_last_element(list_of_lists):
    total = 0
    count = 0
    for sublist in list_of_lists:
        if len(sublist) > 0:  # Ensure sublist is not empty
            total += sublist[-1]
            count += 1
    if count == 0:
        return 0  # Return 0 if there are no valid sublists
    return total / count


# %%
def average_and_std_dev_last_element(list_of_lists, show_mean):
    total = 0
    count = 0
    for sublist in list_of_lists:
        if len(sublist) > 0:  # Ensure sublist is not empty
            total += sublist[-1]
            count += 1
    if count == 0:
        return 0, 0  # Return 0 for both average and standard deviation if there are no valid sublists
    mean = total / count

    # Calculate squared differences and sum them up
    squared_diff_sum = 0
    for sublist in list_of_lists:
        if len(sublist) > 0:  # Ensure sublist is not empty
            squared_diff_sum += (sublist[-1] - mean) ** 2

    # Compute the variance (mean of squared differences)
    variance = squared_diff_sum / count

    # Compute the standard deviation (square root of variance)
    std_deviation = math.sqrt(variance)

    if show_mean:
        return mean
    else:
        return std_deviation


# %%
def average_element(target, input, output):
    total = 0
    count = 0
    for index_list in range(len(input)):
        min_dist = 999999999
        best_output = 9999999999
        input_array = input[index_list]
        output_array = output[index_list]

        for index_inner in range(len(input_array)):
            if np.abs(input_array[index_inner] - target) < min_dist:
                min_dist = np.abs(input_array[index_inner] - target)
                best_output = output_array[index_inner]
        total += best_output
        count += 1

    return total / count


# %%
def average_and_std(target, input, output):
    total = 0
    count = 0
    squared_diff_sum = 0

    best_output_array = []

    for index_list in range(len(input)):
        min_dist = 999999
        best_output = 99999999
        input_array = input[index_list]
        output_array = output[index_list]

        for index_inner in range(len(input_array)):
            if np.abs(input_array[index_inner] - target) < min_dist:
                min_dist = np.abs(input_array[index_inner] - target)
                best_output = output_array[index_inner]

        best_output_array.append(best_output)
        total += best_output
        count += 1

    # Calculate mean
    mean = total / count

    # Calculate squared differences
    for index_list in range(len(input)):
        input_array = input[index_list]
        output_array = output[index_list]

        for best_output_value in best_output_array:
            squared_diff_sum += (best_output_value - mean) ** 2

    # Calculate standard deviation
    std_dev = np.sqrt(squared_diff_sum / count)

    return mean, std_dev


# %%
def get_elements_close(target, input, output):
    count = 0

    best_output_array = []

    for index_list in range(len(input)):
        min_dist = 999999
        best_output = 99999999
        input_array = input[index_list]
        output_array = output[index_list]

        for index_inner in range(len(input_array)):
            if np.abs(input_array[index_inner] - target) < min_dist:
                min_dist = np.abs(input_array[index_inner] - target)
                best_output = output_array[index_inner]

        best_output_array.append(best_output)
        count += 1

    return best_output_array


cp = ModelComparison(filenames_adv)
# cp.assess_initial_performance()
# cp.assess_untarget_performance()
cp.assess_target_performance()

vanilla = ModelComparison(filenames_vanilla, adv=False)
# vanilla.assess_initial_performance()
# vanilla.assess_untarget_performance()
# vanilla.assess_target_performance()

import os
import torch
import shutup
import statistics as s
import numpy as np

import matplotlib.pyplot as plt

from configuration import base_path

shutup.please()


def load_value_from_checkpoint(epoch):
    if os.path.isfile("model_epoch" + str(epoch) + ".pt"):
        checkpoint = torch.load("model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss_shape = checkpoint['training_losses_shape']
        overall_training_loss_height = checkpoint['training_losses_height']
        overall_validation_loss_shape = checkpoint['validation_losses_shape']
        overall_validation_loss_height = checkpoint['validation_losses_height']

        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']

        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']

        overall_training_acc = checkpoint["training_accuracies"]
        overall_validation_acc = checkpoint["validation_accuracies"]

        return overall_training_loss_shape[epoch - 5], overall_validation_loss_shape[epoch - 5], \
               overall_training_loss_height[epoch - 5], overall_validation_loss_height[epoch - 5], \
               overall_training_mse[epoch - 5], overall_validation_mse[epoch - 5], \
               overall_training_ssim[epoch - 5], overall_validation_ssim[epoch - 5], \
               overall_training_acc[epoch - 5], overall_validation_acc[epoch - 5]

    else:
        print("No model found with epoch {}".format(
            str(epoch)
        ))


def load_graphs_from_checkpoint(epoch):
    if os.path.isfile("model_epoch" + str(epoch) + ".pt"):
        checkpoint = torch.load("model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss_shape = checkpoint['training_losses_shape']
        overall_training_loss_height = checkpoint['training_losses_height']
        overall_validation_loss_shape = checkpoint['validation_losses_shape']
        overall_validation_loss_height = checkpoint['validation_losses_height']

        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']

        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']

        overall_training_acc = checkpoint["training_accuracies"]
        overall_validation_acc = checkpoint["validation_accuracies"]

        return overall_training_loss_shape, overall_validation_loss_shape, \
               overall_training_loss_height, overall_validation_loss_height, \
               overall_training_mse, overall_validation_mse, \
               overall_training_ssim, overall_validation_ssim, \
               overall_training_acc, overall_validation_acc

    else:
        print("No model found with epoch {}".format(
            str(epoch)
        ))


def load_graphs_from_checkpoint_combined(epoch):
    if os.path.isfile("model_epoch" + str(epoch) + ".pt"):
        checkpoint = torch.load("model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss_height = checkpoint['training_losses']
        overall_validation_loss_height = checkpoint['validation_losses']

        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']

        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']

        return overall_training_loss_height, overall_validation_loss_height, \
               overall_training_mse, overall_validation_mse, \
               overall_training_ssim, overall_validation_ssim, \

    else:
        print("No model found with epoch {}".format(
            str(epoch)
        ))


def find_max_epoch_combined(path):
    files = os.listdir(path)

    max_epoch = -1

    for file in files:
        if file.__contains__(".pt"):
            epoch = int(file.replace("model_epoch", "").replace(".pt", ""))

            if epoch > max_epoch:
                max_epoch = epoch

    return load_graphs_from_checkpoint_combined(max_epoch)


def find_max_epoch(path, v):
    files = os.listdir(path)

    max_epoch = -1

    for file in files:
        if file.__contains__(".pt"):
            epoch = int(file.replace("model_epoch", "").replace(".pt", ""))

            if epoch > max_epoch:
                max_epoch = epoch

    if not v:
        return load_graphs_from_checkpoint(max_epoch)
    else:
        return load_value_from_checkpoint(max_epoch)


if __name__ == '__main__':
    plt.figure()

    # for m in range(1, 8):
    #     current = "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below" + str(m)
    #
    #     os.chdir(current)
    #
    #     tr_loss_shapes, val_loss_shapes, tr_loss_heights, val_loss_heights, tr_mses, val_mses, \
    #     tr_ssims, val_ssims, tr_accs, val_accs = find_max_epoch(current, False)
    #
    #     plt.plot(tr_ssims, c='blue')
    #
    # current = "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_combined_L1Loss_Adam_UNET_0.001"
    # os.chdir(current)
    #
    # tr_loss_heights, val_loss_heights, tr_mses, val_mses, tr_ssims, val_ssims = find_max_epoch_combined(current)
    #
    # plt.plot(tr_ssims, c='red')
    #
    # plt.legend(loc="lower right", fontsize=10)
    # plt.show()

    points = []

    for m in range(1, 11):
        current = "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below" + str(m)

        os.chdir(current)

        tr_loss_shape, val_loss_shape, tr_loss_height, val_loss_height, tr_mse, val_mse, \
        tr_ssim, val_ssim, tr_acc, val_acc = find_max_epoch(current, True)

        points.append(val_ssim)

    plt.plot(points)
    plt.show()

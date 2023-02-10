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

        return overall_training_loss_shape[epoch - 6], overall_validation_loss_shape[epoch - 6], \
               overall_training_loss_height[epoch - 6], overall_validation_loss_height[epoch - 6], \
               overall_training_mse[epoch - 6], overall_validation_mse[epoch - 6], \
               overall_training_ssim[epoch - 6], overall_validation_ssim[epoch - 6], \
               overall_training_acc[epoch - 6], overall_validation_acc[epoch - 6]

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

        overall_training_f1 = checkpoint["training_f1s"]
        overall_validation_f1 = checkpoint["validation_f1s"]

        overall_training_recall = checkpoint["training_recalls"]
        overall_validation_recall = checkpoint["validation_recalls"]

        overall_training_precision = checkpoint["training_precisions"]
        overall_validation_precision = checkpoint["validation_precisions"]

        overall_training_acc = checkpoint["training_accuracies"]
        overall_validation_acc = checkpoint["validation_accuracies"]

        return overall_training_loss_shape, overall_validation_loss_shape, \
               overall_training_loss_height, overall_validation_loss_height, \
               overall_training_mse, overall_validation_mse, \
               overall_training_ssim, overall_validation_ssim, \
               overall_training_acc, overall_validation_acc, \
               overall_training_f1, overall_validation_f1, \
               overall_training_recall, overall_validation_recall, \
               overall_training_precision, overall_validation_precision

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
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    fig.text(0.5, 0.02, 'Amount of epochs trained', ha='center', size=16.9)
    fig.text(0.07, 0.5, 'Score [0, 1]', va='center', rotation='vertical', size=16.9)

    axs[0].title.set_text('F1-Score')
    axs[1].title.set_text('Accuracy')

    for m in range(1, 11):
        col = (np.random.random(), np.random.random(), np.random.random())
        current = "C:/Users/Konstantin/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below" + str(m)

        os.chdir(current)

        tr_loss_shapes, val_loss_shapes, tr_loss_heights, val_loss_heights, tr_mses, val_mses, \
        tr_ssims, val_ssims, tr_accs, val_accs, tr_f1s, val_f1s, tr_recs, val_recs, tr_precs, val_precs = find_max_epoch(current, False)

        axs[0].plot(
            2 * np.divide(np.multiply(tr_precs, tr_recs), (np.add(tr_precs, tr_recs))),
            c=col,
            label=chr(964) + " < " + str(m) + "m"
        )
        axs[0].legend()
        axs[1].plot(
            tr_accs,
            c=col,
            label=chr(964) + " < " + str(m) + "m"
        )
        axs[1].legend()


    # current = "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_combined_L1Loss_Adam_UNET_1e-05"
    # os.chdir(current)

    # tr_loss_heights, val_loss_heights, tr_mses, val_mses, tr_ssims, val_ssims = find_max_epoch_combined(current)
    # plt.plot(tr_ssims, c='red')

    # plt.legend(loc="lower right", fontsize=10)
    plt.savefig("C:/Users/Konstantin/Desktop/scientific_figures/f1saccs.png", dpi=600)
    plt.close(fig)

    # points = []
    #
    # for m in range(1, 11):
    #     current = "C:/Users/Konstantin/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below" + str(m)
    #
    #     os.chdir(current)
    #
    #     tr_loss_shape, val_loss_shape, tr_loss_height, val_loss_height, tr_mse, val_mse, \
    #     tr_ssim, val_ssim, tr_acc, val_acc = find_max_epoch(current, True)
    #
    #     points.append(val_loss_height)
    #
    # plt.plot(points)
    # plt.show()
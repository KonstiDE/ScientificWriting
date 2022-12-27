import os
import torch
import shutup
import statistics as s
import numpy as np

import matplotlib.pyplot as plt

shutup.please()


def load_graphs_from_checkpoint(epoch):
    if os.path.isfile("../model_epoch" + str(epoch) + ".pt"):
        checkpoint = torch.load("../model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss_shape = checkpoint['training_losses_shape']
        overall_training_loss_height = checkpoint['training_losses_height']
        overall_validation_loss_shape = checkpoint['validation_losses_shape']
        overall_validation_loss_height = checkpoint['validation_losses_height']

        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']
        overall_training_accuracy = checkpoint['training_accuracies']
        overall_training_f1 = checkpoint['training_f1s']
        overall_training_recall = checkpoint['training_recalls']
        overall_training_precision = checkpoint['training_precisions']

        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']
        overall_validation_accuracy = checkpoint['validation_accuracies']
        overall_validation_f1 = checkpoint['validation_f1s']
        overall_validation_recall = checkpoint['validation_recalls']
        overall_validation_precision = checkpoint['validation_precisions']

        plt.figure()
        plt.plot(overall_training_loss_shape, 'b', label="Training loss Shape")
        plt.plot(overall_validation_loss_shape, 'r', label="Validation loss Shape")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_loss_height, 'b', label="Training loss Height")
        plt.plot(overall_validation_loss_height, 'r', label="Validation loss Height")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()


        plt.figure()
        plt.plot(overall_training_mse, 'b', label="Training MSE")
        plt.plot(overall_validation_mse, 'orange', label="Validation MSE")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_ssim, 'b', label="Training SSIM")
        plt.plot(overall_validation_ssim, 'orange', label="Validation SSIM")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_accuracy, 'b', label="Training Accuracy")
        plt.plot(overall_validation_accuracy, 'orange', label="Validation Accuracy")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_f1, 'b', label="Training F1")
        plt.plot(overall_validation_f1, 'orange', label="Validation F1")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_recall, 'b', label="Training Recall")
        plt.plot(overall_validation_recall, 'orange', label="Validation Recall")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_precision, 'b', label="Training Precision")
        plt.plot(overall_validation_precision, 'orange', label="Validation Precision")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        print('MSE train: ' + str(overall_training_mse[epoch - 1]))
        print('MSE valid: ' + str(overall_validation_mse[epoch - 1]))

        print('SSIM train: ' + str(overall_training_ssim[epoch - 1]))
        print('SSIM valid: ' + str(overall_validation_ssim[epoch - 1]))

        print('ACCURACY train: ' + str(overall_training_accuracy[epoch - 1]))
        print('ACCURACY valid: ' + str(overall_validation_accuracy[epoch - 1]))

        print('F1 train: ' + str(overall_training_f1[epoch - 1]))
        print('F1 valid: ' + str(overall_validation_f1[epoch - 1]))

        print('RECALL train: ' + str(overall_training_recall[epoch - 1]))
        print('RECALL valid: ' + str(overall_validation_recall[epoch - 1]))

        print('PRECISION train: ' + str(overall_training_precision[epoch - 1]))
        print('PRECISION valid: ' + str(overall_validation_precision[epoch - 1]))

    else:
        print("No model found with epoch {}".format(
            str(epoch)
        ))


if __name__ == '__main__':
    load_graphs_from_checkpoint(28)

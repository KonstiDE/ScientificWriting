import math

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.metrics as skl
import warnings
from tqdm.auto import tqdm as prog

import torch
import torch.nn as nn

from builder.dataset_provider_combined import get_dataset as get_dataset_combined
from builder.dataset_provider_single import get_dataset as get_dataset_split

from model.model_single_shape import UNET_SHAPE
from model.model_single_height import UNET_HEIGHT

import importlib.util
import sys

warnings.filterwarnings("ignore")

DATA_PATH_COMBINED = "F:/data/combined/test/"
DATA_PATH_SPLIT = "F:/data/single/test/"
MODEL_PATH_COMBINED = "../output/best_of_models/results_combined_L1Loss_Adam_UNET_1e-05/model_epoch28.pt"
MODEL_PATH_SPLIT_TAU1 = "../output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below1/model_epoch24.pt"
MODEL_PATH_SPLIT_TAU4 = "../output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below4/model_epoch41.pt"
MODEL_PATH_SPLIT_TAU10 = "../output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below10/model_epoch25.pt"
BATCH_SIZE = 1
DEVICE = "cuda:0"
px = 1 / plt.rcParams['figure.dpi']


def crop_center(array, crop):
    y, x = array.shape
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    return array[starty:starty + crop, startx:startx + crop]


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def perform_tests(loaders, models, sample_ids=None):
    if sample_ids is None:
        sample_ids = [1, 2]

    for height in range(35, 45):
        fig, axs = plt.subplots(len(sample_ids), 2 + len(models), figsize=(29, height))

        h = 0

        for sample_id in sample_ids:
            first_done = False

            for i in range(len(models)):
                data, target, dataframe_path = loaders[i].__getitem_by_name__(sample_id)

                data = data.cpu()
                target = target.cpu()

                data[data < 0] = 0
                target[target < 0] = 0

                if not first_done:
                    first_done = True

                    data = data.squeeze(0).cpu()
                    red = crop_center(data[0].numpy(), 500)
                    red_normalized = (red * (1 / red.max()))
                    green = crop_center(data[1].numpy(), 500)
                    green_normalized = (green * (1 / green.max()))
                    blue = crop_center(data[2].numpy(), 500)
                    blue_normalized = (blue * (1 / blue.max()))

                    beauty = np.dstack((blue_normalized, green_normalized, red_normalized))

                    target = crop_center(target, 500)

                    im = axs[h, 0].imshow(beauty)
                    axs[h, 0].set_xticklabels([])
                    axs[h, 0].set_yticklabels([])

                    im = axs[h, 1].imshow(target, cmap="viridis")
                    axs[h, 1].set_xticklabels([])
                    axs[h, 1].set_yticklabels([])
                    cbar = plt.colorbar(im, ax=axs[h, 1])
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(26)

                data = data.unsqueeze(0)
                prediction = models[i](data).squeeze(0).squeeze(0).detach().cpu()

                prediction[prediction < 0] = 0

                prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
                target = target.squeeze(0).squeeze(0).detach().cpu()

                target = crop_center(target, 500)
                prediction = crop_center(prediction, 500)

                mae = skl.mean_absolute_error(target, prediction)
                mse = skl.mean_squared_error(target, prediction)

                im = axs[h, 2 + i].imshow(prediction, cmap="viridis")
                axs[h, 2 + i].set_xticklabels([])
                axs[h, 2 + i].set_yticklabels([])
                cbar = plt.colorbar(im, ax=axs[h, 2 + i])
                im.set_clim(0, max(prediction.max(), target.max()))
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(26)
                axs[h, 2 + i].set_xlabel("MAE: {:.2f}\nRMSE: {:.2f}".format(mae, math.sqrt(mse)), fontsize=30)

            h += 1

        plt.tight_layout()
        plt.savefig("/home/fkt48uj/nrw/results/visual_results_{}.png".format(height), dpi=400)


def setup():
    test_loader_combined = get_dataset_combined(DATA_PATH_COMBINED, percentage_load=1)
    test_loader_split_tau1 = get_dataset_split(DATA_PATH_SPLIT, percentage_load=1, below_m=1)
    test_loader_split_tau4 = get_dataset_split(DATA_PATH_SPLIT, percentage_load=1, below_m=4)
    test_loader_split_tau10 = get_dataset_split(DATA_PATH_SPLIT, percentage_load=1, below_m=10)

    model_combined = UNET_SHAPE()
    model_split_tau1 = UNET_HEIGHT()
    model_split_tau4 = UNET_HEIGHT()
    model_split_tau10 = UNET_HEIGHT()

    model_combined.load_state_dict(torch.load(MODEL_PATH_COMBINED, map_location='cpu')['model_state_dict'])
    model_split_tau1.load_state_dict(torch.load(MODEL_PATH_SPLIT_TAU1, map_location='cpu')['model_state_dict'])
    model_split_tau4.load_state_dict(torch.load(MODEL_PATH_SPLIT_TAU4, map_location='cpu')['model_state_dict'])
    model_split_tau10.load_state_dict(torch.load(MODEL_PATH_SPLIT_TAU10, map_location='cpu')['model_state_dict'])

    model_combined.eval()
    model_split_tau1.eval()
    model_split_tau4.eval()
    model_split_tau10.eval()

    perform_tests(
        [test_loader_combined, test_loader_split_tau1, test_loader_split_tau4, test_loader_split_tau10],
        [model_combined, model_split_tau1, model_split_tau4, model_split_tau10],
        [
            #Tiles to test for matrix
            ""
        ]
    )


if __name__ == '__main__':
    setup()

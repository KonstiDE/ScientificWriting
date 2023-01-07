import os.path

import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm as prog
from PIL import Image


from builder.dataset_provider_single import get_loader

from configuration import (
    pin_memory,
    num_workers,
    device,
    base_path
)

from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError
)

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision
)

from torchmetrics.image import StructuralSimilarityIndexMeasure

from model.model_single_shape import UNET_SHAPE
from model.model_single_height import UNET_HEIGHT

import torchvision.transforms.functional as tf

import sys
sys.path.append(os.getcwd())

import shutup
shutup.please()


def test(model_path, test_data_path, m):
    model_shape = UNET_SHAPE(in_channels=4, out_channels=1)
    model_height = UNET_HEIGHT(in_channels=4, out_channels=1)

    model_shape.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict_shape'])
    model_height.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict_height'])
    model_shape.to(device)
    model_height.to(device)

    model_shape.eval()
    model_height.eval()

    torch.no_grad()

    loader = get_loader(
        npz_dir=test_data_path,
        batch_size=1,
        percentage_load=1,
        pin_memory=pin_memory,
        num_workers=num_workers,
        below_m=m,
        shuffle=True
    )
    c = 0
    
    loop = prog(loader)

    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    accuracy = BinaryAccuracy().to(device)
    f1 = BinaryF1Score().to(device)
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)

    running_mae = []
    running_mse = []
    running_ssim = []

    running_accuracy = []
    running_f1 = []
    running_recall = []
    running_precision = []

    for batch_index, (data, target_shape, target_height) in enumerate(loop):
        data = data.to(device)

        target_shape = target_shape.to(device).unsqueeze(1)
        target_height = target_height.to(device).unsqueeze(1)

        target_shape = torch.round(target_shape)

        with torch.no_grad():
            prediction_shape = model_shape(data)
            prediction_height = model_height(data, prediction_shape.detach().clone())

        running_accuracy.append(accuracy(prediction_shape, target_shape).item())
        running_f1.append(f1(prediction_shape, target_shape).item())
        running_recall.append(recall(prediction_shape, target_shape).item())
        running_precision.append(precision(prediction_shape, target_shape).item())

        running_mae.append(mae(prediction_height, target_height).item())
        running_mse.append(mse(prediction_height, target_height).item())
        running_ssim.append(ssim(prediction_height, target_height).item())

        accuracy.reset()
        f1.reset()
        recall.reset()
        precision.reset()

        mae.reset()
        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Progress {}".format(""))

        prediction_shape = torch.round(prediction_shape)

        prediction_shape = prediction_shape.squeeze(0).squeeze(0).detach().cpu()
        target_shape = target_shape.squeeze(0).squeeze(0).detach().cpu()

        prediction_height = prediction_height.squeeze(0).squeeze(0).detach().cpu()
        target_height = target_height.squeeze(0).squeeze(0).detach().cpu()

        data = data.squeeze(0).cpu().numpy()
        red = data[0]
        red_normalized = (red * (1 / red.max()))
        green = data[1]
        green_normalized = (green * (1 / green.max()))
        blue = data[2]
        blue_normalized = (blue * (1 / blue.max()))

        beauty = np.dstack((blue_normalized, green_normalized, red_normalized))

        fig, axs = plt.subplots(1, 5, figsize=(28, 5))

        im = axs[0].imshow(beauty)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

        im = axs[1].imshow(prediction_shape, cmap="Greys")
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(target_shape, cmap="Greys")
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        plt.colorbar(im, ax=axs[2])

        im = axs[3].imshow(prediction_height, cmap="viridis")
        axs[3].set_xticklabels([])
        axs[3].set_yticklabels([])
        plt.colorbar(im, ax=axs[3])

        im = axs[4].imshow(target_height, cmap="viridis")
        axs[4].set_xticklabels([])
        axs[4].set_yticklabels([])
        plt.colorbar(im, ax=axs[4])

        fig.suptitle("ACC: {:.3f}, F1: {:.3f}, REC: {:.3f}, PREC: {:.3f}, MAE: {:.3f}, MSE: {:.3f}, SSIM: {:.3f}".format(
            running_accuracy[-1],
            running_f1[-1],
            running_recall[-1],
            running_precision[-1],
            running_mae[-1],
            running_mse[-1],
            running_ssim[-1]
        ), fontsize=20)

        plt.savefig("B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below" + str(m) + "/results/" + str(c) + ".png")
        plt.close(fig)

        c += 1


    file = open("B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below" + str(m) + "/results/results.txt", "w+")
    file.write("ACC: {}, F1: {}, REC: {}, PREC: {}, MAE: {}, MSE: {}, SSIM: {}".format(
        str(s.mean(running_accuracy)),
        str(s.mean(running_f1)),
        str(s.mean(running_recall)),
        str(s.mean(running_precision)),
        str(s.mean(running_mae)),
        str(s.mean(running_mse)),
        str(s.mean(running_ssim)),
    ))
    file.close()


if __name__ == '__main__':

    models = [
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below1/model_epoch38.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below2/model_epoch22.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below3/model_epoch26.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below4/model_epoch22.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below5/model_epoch24.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below6/model_epoch29.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below7/model_epoch23.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below8/model_epoch20.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below9/model_epoch29.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_single_BCEWithLogitsLoss_L1Loss_Adam_Adam_UNET_SHAPE_UNET_HEIGHT_1e-05_1e-05_below10/model_epoch37.pt",
    ]

    m = 1
    for model in models:
        test(
            model,
            "B:/projects/PycharmProjects/ScientificWriting/output/single_belows/test/",
            m
        )
        m += 1

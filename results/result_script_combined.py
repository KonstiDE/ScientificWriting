import os.path

import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm as prog
from PIL import Image


from builder.dataset_provider_combined import get_loader

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

from torchmetrics.image import StructuralSimilarityIndexMeasure

from model.model_combined import UNET

import torchvision.transforms.functional as tf

import sys
sys.path.append(os.getcwd())

import shutup
shutup.please()


def test(model_path, test_data_path):
    model = UNET(in_channels=4, out_channels=1)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    model.to(device)

    model.eval()

    torch.no_grad()

    loader = get_loader(
        npz_dir=test_data_path,
        batch_size=1,
        percentage_load=1,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False
    )
    c = 0
    
    loop = prog(loader)

    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    running_mae = []
    running_mse = []
    running_ssim = []

    for batch_index, (name, data, target) in enumerate(loop):
        data = data.to(device)

        target = target.to(device).unsqueeze(1)

        with torch.no_grad():
            prediction = model(data)

        running_mae.append(mae(prediction, target).item())
        running_mse.append(mse(prediction, target).item())
        running_ssim.append(ssim(prediction, target).item())

        mae.reset()
        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Progress {}".format(""))

        prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
        target = target.squeeze(0).squeeze(0).detach().cpu()

        data = data.squeeze(0).cpu().numpy()
        red = data[0]
        red_normalized = (red * (1 / red.max()))
        green = data[1]
        green_normalized = (green * (1 / green.max()))
        blue = data[2]
        blue_normalized = (blue * (1 / blue.max()))

        beauty = np.dstack((blue_normalized, green_normalized, red_normalized))

        fig, axs = plt.subplots(1, 4, figsize=(27, 5))

        im = axs[0].imshow(beauty)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

        im = axs[1].imshow(prediction, cmap="viridis")
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        im.set_clim(0, max(prediction.max(), target.max()))
        plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(target, cmap="viridis")
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        im.set_clim(0, max(prediction.max(), target.max()))
        plt.colorbar(im, ax=axs[2])

        im = axs[3].imshow(abs(target - prediction), cmap="turbo")
        axs[3].set_xticklabels([])
        axs[3].set_yticklabels([])
        plt.colorbar(im, ax=axs[3])

        fig.suptitle("MAE: {:.3f}, MSE: {:.3f}, SSIM: {:.3f}".format(
            running_mae[-1],
            running_mse[-1],
            running_ssim[-1]
        ), fontsize=20)

        plt.savefig("B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_combined_L1Loss_Adam_UNET_0.001/results/" + os.path.basename(name[0]) + ".png")
        plt.close(fig)

        c += 1

    file = open("B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_combined_L1Loss_Adam_UNET_0.001/results/results.txt", "w+")
    file.write("MAE: {}, MSE: {}, SSIM: {}".format(
        str(s.mean(running_mae)),
        str(s.mean(running_mse)),
        str(s.mean(running_ssim)),
    ))
    file.close()


if __name__ == '__main__':
    test(
        "B:/projects/PycharmProjects/ScientificWriting/output/best_of_models/results_combined_L1Loss_Adam_UNET_0.001/model_epoch32.pt",
        "B:/projects/PycharmProjects/ScientificWriting/output/combined/test/"
    )

import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

import shutup
import torch

shutup.please()

from rasterio.plot import show


def view_data_frame(path):
    files = os.listdir(path)

    for file in files:
        data_frame = np.load(os.path.join(path, file), allow_pickle=True)

        red = data_frame["red"]
        green = data_frame["green"]
        blue = data_frame["blue"]

        below1 = data_frame["dsm_below1"]
        below2 = data_frame["dsm_below2"]
        below3 = data_frame["dsm_below3"]
        below4 = data_frame["dsm_below4"]
        below5 = data_frame["dsm_below5"]
        below6 = data_frame["dsm_below6"]
        below7 = data_frame["dsm_below7"]
        below8 = data_frame["dsm_below8"]
        below9 = data_frame["dsm_below9"]
        below10 = data_frame["dsm_below10"]

        dsm = data_frame["dsm_og"]

        assert (below1[np.where(below1 == 0)].size + below1[np.where(below1 == 1)].size == 512*512)
        assert (below2[np.where(below2 == 0)].size + below2[np.where(below2 == 1)].size == 512 * 512)
        assert (below3[np.where(below3 == 0)].size + below3[np.where(below3 == 1)].size == 512 * 512)
        assert (below4[np.where(below4 == 0)].size + below4[np.where(below4 == 1)].size == 512 * 512)
        assert (below5[np.where(below5 == 0)].size + below5[np.where(below5 == 1)].size == 512 * 512)
        assert (below6[np.where(below6 == 0)].size + below6[np.where(below6 == 1)].size == 512 * 512)
        assert (below7[np.where(below7 == 0)].size + below7[np.where(below7 == 1)].size == 512 * 512)
        assert (below8[np.where(below8 == 0)].size + below8[np.where(below8 == 1)].size == 512 * 512)
        assert (below9[np.where(below9 == 0)].size + below9[np.where(below9 == 1)].size == 512 * 512)
        assert (below10[np.where(below10 == 0)].size + below10[np.where(below10 == 1)].size == 512 * 512)

        fig, axs = plt.subplots(4, 3, figsize=(14, 18))

        for i in range(4):
            for t in range(3):
                axs[i][t].set_xticklabels([])
                axs[i][t].set_yticklabels([])

        red_normalized = (red * (1 / red.max()))
        green_normalized = (green * (1 / green.max()))
        blue_normalized = (blue * (1 / blue.max()))

        beauty = np.dstack((blue_normalized, green_normalized, red_normalized))

        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])

        fig.patch.set_facecolor('white')

        axs[0][0].imshow(beauty)
        axs[0][1].imshow(dsm, cmap='viridis')
        axs[0][2].imshow(below1, cmap='copper')
        axs[1][0].imshow(below2, cmap='copper')
        axs[1][1].imshow(below3, cmap='copper')
        axs[1][2].imshow(below4, cmap='copper')
        axs[2][0].imshow(below5, cmap='copper')
        axs[2][1].imshow(below6, cmap='copper')
        axs[2][2].imshow(below7, cmap='copper')
        axs[3][0].imshow(below8, cmap='copper')
        axs[3][1].imshow(below9, cmap='copper')
        axs[3][2].imshow(below10, cmap='copper')

        plt.tight_layout(pad=2)
        plt.savefig(str.format("../output/{}.png", file))
        plt.close(fig)

    else:
        print("{} is not a valid data frame".format(path))


if __name__ == '__main__':
    view_data_frame("../output/single_belows/test/")

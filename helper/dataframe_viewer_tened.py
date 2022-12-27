import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

import shutup
import torch

shutup.please()

from rasterio.plot import show


def view_data_frame(path):
    if path.endswith(".npz"):
        data_frame = np.load(path, allow_pickle=True)

        red = data_frame["red"]
        green = data_frame["green"]
        blue = data_frame["blue"]
        nir = data_frame["nir"]

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

        plt.imshow(red, cmap='Reds_r')
        plt.colorbar()
        plt.title("Red")
        plt.show()

        plt.imshow(green, cmap='Greens_r')
        plt.colorbar()
        plt.title("Green")
        plt.show()

        plt.imshow(blue, cmap='Blues_r')
        plt.colorbar()
        plt.title("Blue")
        plt.show()

        plt.imshow(nir, cmap='Purples_r')
        plt.colorbar()
        plt.title("Near Infrared")
        plt.show()

        plt.imshow(below1, cmap='Greys_r')
        plt.title("GT < 1")
        plt.colorbar()
        plt.show()

        plt.imshow(below2, cmap='Greys_r')
        plt.title("GT < 2")
        plt.colorbar()
        plt.show()

        plt.imshow(below3, cmap='Greys_r')
        plt.title("GT < 3")
        plt.colorbar()
        plt.show()

        plt.imshow(below4, cmap='Greys_r')
        plt.title("GT < 4")
        plt.colorbar()
        plt.show()

        plt.imshow(below5, cmap='Greys_r')
        plt.title("GT < 5")
        plt.colorbar()
        plt.show()

        plt.imshow(below6, cmap='Greys_r')
        plt.title("GT < 6")
        plt.colorbar()
        plt.show()

        plt.imshow(below7, cmap='Greys_r')
        plt.title("GT < 7")
        plt.colorbar()
        plt.show()

        plt.imshow(below8, cmap='Greys_r')
        plt.title("GT < 8")
        plt.colorbar()
        plt.show()

        plt.imshow(below9, cmap='Greys_r')
        plt.title("GT < 9")
        plt.colorbar()
        plt.show()

        plt.imshow(below10, cmap='Greys_r')
        plt.title("GT < 10")
        plt.colorbar()
        plt.show()

    else:
        print("{} is not a valid data frame".format(path))


if __name__ == '__main__':
    os.chdir("C:/Users/s371513/Desktop/ScientificWriting/ScientificWriting")

    view_data_frame("output/width200_height0.npz")

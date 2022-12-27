import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

import shutup
shutup.please()

from rasterio.plot import show


def view_data_frame(path):
    if path.endswith(".npz"):
        data_frame = np.load(path, allow_pickle=True)

        red = data_frame["red"]
        green = data_frame["green"]
        blue = data_frame["blue"]
        nir = data_frame["nir"]
        dsm = data_frame["dsm"]

        print(blue.shape)
        print(green.shape)
        print(red.shape)
        print(nir.shape)
        print(dsm.shape)

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

        plt.imshow(dsm, cmap='viridis')
        plt.title("nDSM")
        plt.colorbar()
        plt.show()

    else:
        print("{} is not a valid data frame".format(path))


if __name__ == '__main__':
    os.chdir("B:/projects/PycharmProjects/ScientificWriting")

    view_data_frame("output/width0_height0.npz")

import os

import numpy as np
import rasterio as rio
from rasterio.windows import Window

np.seterr(divide='ignore', invalid='ignore')


def createAppendix():
    band_red_filename = "F:/data/sentinel2_red_clip_resaBiLin.tif"
    band_green_filename = "F:/data/sentinel2_green_clip_resaBiLin.tif"
    band_blue_filename = "F:/data/sentinel2_blue_clip_resaBiLin.tif"
    band_nir_filename = "F:/data/sentinel2_nir_clip_resaBiLin.tif"
    dsm_filename = "F:/data/nDSM_Muc.tif"

    dsm_filenames = [
        "F:/data_extended/nDSM_Muc_gt_below1.tif",
        "F:/data_extended/nDSM_Muc_gt_below2.tif",
        "F:/data_extended/nDSM_Muc_gt_below3.tif",
        "F:/data_extended/nDSM_Muc_gt_below4.tif",
        "F:/data_extended/nDSM_Muc_gt_below5.tif",
        "F:/data_extended/nDSM_Muc_gt_below6.tif",
        "F:/data_extended/nDSM_Muc_gt_below7.tif",
        "F:/data_extended/nDSM_Muc_gt_below8.tif",
        "F:/data_extended/nDSM_Muc_gt_below9.tif",
        "F:/data_extended/nDSM_Muc_gt_below10.tif",
    ]

    dsm_rios = [
        rio.open(os.path.join("data", dsm_filenames[0])),
        rio.open(os.path.join("data", dsm_filenames[1])),
        rio.open(os.path.join("data", dsm_filenames[2])),
        rio.open(os.path.join("data", dsm_filenames[3])),
        rio.open(os.path.join("data", dsm_filenames[4])),
        rio.open(os.path.join("data", dsm_filenames[5])),
        rio.open(os.path.join("data", dsm_filenames[6])),
        rio.open(os.path.join("data", dsm_filenames[7])),
        rio.open(os.path.join("data", dsm_filenames[8])),
        rio.open(os.path.join("data", dsm_filenames[9]))
    ]

    band_red_path = os.path.join("data", band_red_filename)
    band_green_path = os.path.join("data", band_green_filename)
    band_blue_path = os.path.join("data", band_blue_filename)
    band_nir_path = os.path.join("data", band_nir_filename)
    dsm_path = os.path.join("data", dsm_filename)

    red = rio.open(band_red_path)
    green = rio.open(band_green_path)
    blue = rio.open(band_blue_path)
    nir = rio.open(band_nir_path)
    dsm = rio.open(dsm_path)

    # swap: width<->height
    no_data_value = -3.4028234663852886e+38

    WINDOW_SIZE = 512
    i = 0
    # Manual tested offset
    t = 1500
    b = 0
    c = 0

    i, t = get_max_offsets(os.path.join("output", "single_belows"))
    print(i)
    print(t)

    while t < dsm.height:
        while i < dsm.width:
            window = Window(i, t, WINDOW_SIZE, WINDOW_SIZE)

            data_dsm_og = dsm.read(1, window=window)

            if no_data_value not in data_dsm_og and len(data_dsm_og) != 0:
                data_red = red.read(1, window=window)
                data_green = green.read(1, window=window)
                data_blue = blue.read(1, window=window)
                data_nir = nir.read(1, window=window)
                data_dsm_og[data_dsm_og < 0] = 0

                cropping_list = []

                for k in range(10):
                    data_dsm = dsm_rios[k].read(1, window=window)
                    data_dsm[data_dsm < 0] = 0

                    cropping_list.append(data_dsm)

                np.savez_compressed(
                    str.format("output/single_belows/{}.npz", str(window).replace(" ", "")),
                    red=data_red,
                    green=data_green,
                    blue=data_blue,
                    nir=data_nir,
                    dsm_below1=cropping_list[0],
                    dsm_below2=cropping_list[1],
                    dsm_below3=cropping_list[2],
                    dsm_below4=cropping_list[3],
                    dsm_below5=cropping_list[4],
                    dsm_below6=cropping_list[5],
                    dsm_below7=cropping_list[6],
                    dsm_below8=cropping_list[7],
                    dsm_below9=cropping_list[8],
                    dsm_below10=cropping_list[9],
                    dsm_og=data_dsm_og
                )
                print("Successfully builded " + str(window))

                b += 1
                i += WINDOW_SIZE

            else:
                i += 1

        i = 0
        c += 1
        t += WINDOW_SIZE


def get_max_offsets(path):
    files = os.listdir(path)

    max_col_offset = 0
    max_row_offset = 0

    for file in files:
        if file.__contains__(".npz"):
            split = file.split("=")

            col_offset = int(split[1].replace(",row_off", ""))
            row_offset = int(split[2].replace(",width", ""))

            if col_offset >= max_col_offset and row_offset >= max_row_offset:
                max_col_offset = col_offset
                max_row_offset = row_offset

    return max_col_offset, max_row_offset


if __name__ == '__main__':
    os.chdir("B:/projects/PycharmProjects/ScientificWriting")

    createAppendix()

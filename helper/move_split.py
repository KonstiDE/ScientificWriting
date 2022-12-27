import os
import random
import sys

import shutil

sys.path.append(os.getcwd())

from configuration import base_path


def move_into_split():
    split = {
        'train': (0.6, "output/combined/train/", "output/single_belows/train/"),
        'validation': (0.3, "output/combined/validation/", "output/single_belows/validation/"),
        'test': (0.1, "output/combined/test/", "output/single_belows/test/")
    }

    list_of_files = os.listdir("output/combined")
    list_of_files = [filename for filename in list_of_files if '.npz' in filename]
    files_amount = len(list_of_files)

    for split_type, split_tuple in split.items():
        if not os.path.isdir(split_tuple[1]):
            print(split_tuple[1] + " does not exist")
            return

    for split_type, split_tuple in split.items():
        split_percentage = split_tuple[0]
        split_path_combined = split_tuple[1]
        split_path_single = split_tuple[2]

        max_moving = int(files_amount * split_percentage)

        for counter in range(0, max_moving):
            data_frame = random.choice(list_of_files)

            if ".npz" in data_frame:
                shutil.move(
                    os.path.join("output/combined/", data_frame),
                    os.path.join(split_path_combined, data_frame)
                )
                shutil.move(
                    os.path.join("output/single_belows/", data_frame),
                    os.path.join(split_path_single, data_frame)
                )
                list_of_files.remove(data_frame)


if __name__ == '__main__':
    os.chdir(base_path)

    move_into_split()

#!/usr/bin/env python3

import splitfolders
import os
import re
from PIL import Image
# real_train="./real/train/train"
# sim_train="./sim/train/train"
# real_val="./real/train/test"
# sim_val="./sim/train/test"
# real_test="./sim/test"
# sim_test="./sim/test"

# real_data_raw = "./real_images/real"
# sim_data_raw = "./sim_images/sim"


def train_test_split(input_folder, output_folder):
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, 0.1,0.1))

def remove_files():
    train_real = "/sim2real_quarter/train/images_a"
    test_real = "/sim2real_quarter/test/images_a"
    train_sim = "/sim2real_quarter/train/images_b"
    test_sim = "/sim2real_quarter/test/images_b"

    filepath = os.getcwd()
    files_train_real = os.listdir(filepath + train_real)
    files_test_real = os.listdir(filepath + test_real)
    files_train_sim = os.listdir(filepath + train_sim)
    files_test_sim = os.listdir(filepath + test_sim)
    files_train_real.sort()
    files_test_real.sort()
    files_test_sim.sort()
    files_train_sim.sort()
    for file in files_train_real:
        os.remove(filepath+ train_real + "/" +  file)
    for file in files_test_real:
        os.remove(filepath+ test_real + "/" +  file)
    for file in files_train_sim:
        os.remove(filepath+ train_sim + "/" + file)
    for file in files_test_sim:
        os.remove(filepath+ test_sim + "/" + file)

def rename_files():
    train_real = "/sim2real_quarter/train/images_a"
    test_real = "/sim2real_quarter/test/images_a"
    train_sim = "/sim2real_quarter/train/images_b"
    test_sim = "/sim2real_quarter/test/images_b"

    filepath = os.getcwd()
    files_train_real = os.listdir(filepath + train_real)
    files_test_real = os.listdir(filepath + test_real)
    files_train_sim = os.listdir(filepath + train_sim)
    files_test_sim = os.listdir(filepath + test_sim)
    files_train_real.sort()
    files_test_real.sort()
    files_test_sim.sort()
    files_train_sim.sort()
    
    for file in files_train_real:
        os.rename(filepath+ train_real + "/" +  file)
    for file in files_test_real:
        os.rename(filepath+ test_real + "/" +  file)
    for file in files_train_sim:
        os.rename(filepath+ train_sim + "/" + file)
    for file in files_test_sim:
        os.rename(filepath+ test_sim + "/" + file)


if __name__ == '__main__':
    input_dataset = "./dataset_input"
    output_dataset = "./outout_dataset/"
    train_test_split(input_folder=input_dataset, output_folder=output_dataset)
    rename_files()
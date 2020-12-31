#!/usr/bin/env python3

import splitfolders
import os
import re
from PIL import Image

INPUT_DATASET = "./dataset_raw"
OUTPUT_DATASET = "./dataset_output"

real="real"
sim="sim"
TRAIN_REAL= f"{OUTPUT_DATASET}/train/"[1:]
TEST_REAL = f"{OUTPUT_DATASET}/test/"[1:]
TRAIN_SIM = f"{OUTPUT_DATASET}/train/"[1:]
TEST_SIM = f"{OUTPUT_DATASET}/test/"[1:]

def train_test_split(input_folder, output_folder):
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1))


def rename_files_unit_format():

    filepath = os.getcwd()
    files_train_real = os.listdir(f"{filepath}{TRAIN_REAL}{real}")
    files_test_real = os.listdir(f"{filepath}{TEST_REAL}{real}")
    files_train_sim = os.listdir(f"{filepath}{TRAIN_SIM}{sim}")
    files_test_sim = os.listdir(f"{filepath}{TEST_SIM}{sim}")
    files_train_real.sort()
    files_test_real.sort()
    files_test_sim.sort()
    files_train_sim.sort()
    
    for file in files_train_real:
        old_name = f"{filepath}{TRAIN_REAL}{real}/{file}"
        new_name = f"{filepath}{TRAIN_REAL}images_a/{file}"
        os.makedirs(os.path.dirname(f"{filepath}{TRAIN_REAL}images_a/"), exist_ok=True)
        os.rename(old_name, new_name)
    os.rmdir(f"{filepath}{TRAIN_REAL}{real}/")
    for file in files_test_real:
        old_name = f"{filepath}{TEST_REAL}{real}/{file}"
        new_name = f"{filepath}{TEST_REAL}images_a/{file}"
        os.makedirs(os.path.dirname(f"{filepath}{TEST_REAL}images_a/"), exist_ok=True)
        os.rename(old_name, new_name)
    os.rmdir(f"{filepath}{TEST_REAL}{real}/")
    for file in files_train_sim:
        old_name = f"{filepath}{TRAIN_SIM}{sim}/{file}"
        new_name = f"{filepath}{TRAIN_SIM}images_b/{file}"
        os.makedirs(os.path.dirname(f"{filepath}{TRAIN_SIM}images_b/"), exist_ok=True)
        os.rename(old_name, new_name)
    os.rmdir(f"{filepath}{TRAIN_SIM}{sim}/")
    for file in files_test_sim:
        old_name = f"{filepath}{TEST_SIM}{sim}/{file}"
        new_name = f"{filepath}{TEST_SIM}images_b/{file}"
        os.makedirs(os.path.dirname(f"{filepath}{TEST_SIM}images_b/"), exist_ok=True)
        os.rename(old_name, new_name)
    os.rmdir(f"{filepath}{TEST_SIM}{sim}/")


if __name__ == '__main__':
    train_test_split(input_folder=INPUT_DATASET, output_folder=OUTPUT_DATASET)
    rename_files_unit_format()
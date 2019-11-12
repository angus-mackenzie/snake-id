#!/usr/bin/env python

import random
import json
import numpy as np
import argparse
import base64

import time
import traceback

import glob
import os
import json
import dotenv

# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #

from PIL import Image

dotenv.load_dotenv(dotenv.find_dotenv())

DATASET_DIR          = os.environ.setdefault('DATASET_DIR', 'data') # pretty much only need to change this one
DATASET_SSIC_CLASSES = os.environ.setdefault('DATASET_SSIC_CLASSES', os.path.join(DATASET_DIR, 'class_idx_mapping.csv'))
DATASET_SSIC_TRAIN   = os.environ.setdefault('DATASET_SSIC_TRAIN', os.path.join(DATASET_DIR, 'train'))
DATASET_SSIC_TEST    = os.environ.setdefault('DATASET_SSIC_TEST', os.path.join(DATASET_DIR, 'round1'))
OUTPUT_FOLDER        = os.environ.setdefault('OUTPUT_FOLDER', 'out')

# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #


print(os.listdir(DATASET_SSIC_TRAIN)[:10])
print(os.listdir(DATASET_SSIC_TEST)[:10])

# CLASSES

def get_name_class_pairs():
    import pandas as pd
    return [tuple(pair) for pair in pd.read_csv(DATASET_SSIC_CLASSES).values]  # .to_dict('records')

def get_name_cls_map():
    return {name: cls for name, cls in get_name_class_pairs()}

def get_cls_name_map():
    return {cls: name for name, cls in get_name_class_pairs()}


def get_train_imgs():
    img_paths_valid, img_paths_invalid = [], []
    # LOOP THROUGH CLASS FOLDERS
    for cls_name in os.listdir(DATASET_SSIC_TRAIN):
        cls_path = os.path.join(DATASET_SSIC_TRAIN, cls_name)
        # LOOP THROUGH CLASS IMAGES
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            data = (img_path, int(cls_name[len('class-'):]))
            try:
                img = Image.open(img_path)
                img.verify()
                img_paths_valid.append(data)
            except (IOError, SyntaxError) as e:
                img_paths_invalid.append(data)
    return img_paths_valid, img_paths_invalid




img_paths_valid, img_paths_invalid  = get_train_imgs()

print(len(img_paths_valid))
print(len(img_paths_invalid))
print(img_paths_valid[:10])













#
# os.environ['AICROWD_TEST_IMAGES_PATH'] = '/home/nmichlo/downloads/datasets/ssic'
# os.environ['AICROWD_PREDICTIONS_OUTPUT_PATH'] = ''
#
#
# """
# Expected ENVIRONMENT Variables
#
# * AICROWD_TEST_IMAGES_PATH : abs path to  folder containing all the test images
# * AICROWD_PREDICTIONS_OUTPUT_PATH : path where you are supposed to write the output predictions.csv
# """
#
# def gather_images(test_images_path):
#     images = glob.glob(os.path.join(
#         test_images_path, "*.jpg"
#     ))
#     return images
#
# def gather_image_names():
#     images = gather_images(DATASET_SSIC_TEST)
#     image_names = [os.path.basename(image_path) for image_path in images]
#     return image_names
#
# def get_image_path(image_name):
#     test_images_path = os.getenv("AICROWD_TEST_IMAGES_PATH", False)
#     return os.path.join(test_images_path, image_name)
#
# def get_snake_classes():
#     with open('data/class_idx_mapping.csv') as f:
#         classes = []
#         for line in f.readlines()[1:]:
#             class_name = line.split(",")[0]
#             classes.append(class_name)
#     return classes
#
#
# def run():
#     ########################################################################
#     # Gather Image Names
#     ########################################################################
#
#     image_names = gather_image_names()
#
#     ########################################################################
#     # Do your magic here to train the model
#     ########################################################################
#     classes = get_snake_classes()
#
#     def softmax(x):
#         """Compute softmax values for each sets of scores in x."""
#         e_x = np.exp(x - np.max(x))
#         return e_x / e_x.sum(axis=0)
#
#     ########################################################################
#     # Generate Predictions
#     ########################################################################
#
#     LINES = []
#     LINES.append(','.join(['filename'] + classes))
#     predictions = []
#     for image_name in image_names:
#         probs = softmax(np.random.rand(45))
#         probs = list(map(str, probs))
#         LINES.append(",".join([image_name] + probs))
#
#         ########################################################################
#         # Register Prediction
#         #
#         # Note, this prediction register is not a requirement. It is used to
#         # provide you feedback of how far are you in the overall evaluation.
#         # In the absence of it, the evaluation will still work, but you
#         # will see progress of the evaluation as 0 until it is complete
#         #
#         # Here you simply announce that you completed processing a set of
#         # image_names
#         ########################################################################
#         # aicrowd_helpers.execution_progress({
#         #     "image_names" : [image_name]
#         # })
#
#     # Write output
#     fp = open(predictions_output_path, "w")
#     fp.write("\n".join(LINES))
#     fp.close()
#
#     ########################################################################
#     # Register Prediction Complete
#     ########################################################################
#     # aicrowd_helpers.execution_success({
#     #     "predictions_output_path" : predictions_output_path
#     # })
#
#
# if __name__ == "__main__":
#     try:
#         run()
#     except Exception as e:
#         error = traceback.format_exc()
#         print(error)
#         # aicrowd_helpers.execution_error(error)

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from lynx_id.data.triplets import LynxDataset, collate

from megadetector.MegaDetector.detection.run_detector_batch import \
    batch_detection

# This file shows how megadetector can be used with our lynx dataset via a dataloader
# directly in python code.

# Calling MegaDetector in Python code from a notebook does not work.
# The environment variables are not up-to-date.

# Loading the dataset
lynxDataset = LynxDataset(
    Path("/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_france.csv"))

# Defining a dataloader
batch_size = 1024
dataloader = DataLoader(lynxDataset, batch_size=batch_size, shuffle=False, num_workers=1,
                        collate_fn=collate)

# Getting images from the first batch
inputs, outputs = next(iter(dataloader))
images = inputs['images']

# Running MegaDetector
with open("./data_pipeline/megadetector/config_megadetector.json") as f:
    megadetector_args = json.load(f)
bounding_boxes = batch_detection(argparse.Namespace(**megadetector_args), images)

# Flattening the resulting bounding boxes
# For images containing several lynxes, only the first bounding box is taken.
flat_data = [{'file': img['file'],
              'category': img['detections'][0]['category'],
              'conf':img['detections'][0]['conf'],
              'x':img['detections'][0]['bbox'][0],
              'y':img['detections'][0]['bbox'][1],
              'width':img['detections'][0]['bbox'][2],
              'height':img['detections'][0]['bbox'][3]}
             for img in bounding_boxes]
df_bounding_boxes = pd.DataFrame(flat_data)

# Selecting an image from the batch for cropping
selected_image_index = random.randint(0, batch_size-1)
image_i = images[selected_image_index]
x, y, width, height = df_bounding_boxes.iloc[selected_image_index][
    ['x', 'y', 'width', 'height']]

image_i_pil = Image.fromarray(image_i)
im_width, im_height = image_i_pil.size

# Convert the coordinates of the relative bouding boxes
x_norm = x * im_width
y_norm = y * im_height
w_norm = width * im_width
h_norm = height * im_height

left = x_norm
top = y_norm
right = x_norm + w_norm
bottom = y_norm + h_norm

# Crop and save to visualize it
image_i_pil_crop = image_i_pil.crop((left, top, right, bottom))
image_i_pil_crop.save("test.jpg")

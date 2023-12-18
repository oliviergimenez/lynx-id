import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from segment_anything import SamPredictor, sam_model_registry

sys.path.append("../")

from data_pipeline.dataset import LynxDataset
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default="/gpfsscratch/rech/ads/commun/datasets/extracted"
                                                    "/lynx_dataset_full.csv")
parser.add_argument('--save_img_directory', type=str, default="/gpfsscratch/rech/ads/commun/datasets/extracted"
                                                              "/no_background")
args = parser.parse_args()


# Load the dataset from csv
lynxDataset = LynxDataset(Path(args.csv_file))

csv = pd.read_csv(args.csv_file)
# All paths to images without backgrounds to add them to the csv
all_filepath_no_bg = []

# Load and init Segment Anything Model (SAM) on GPU
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint="/gpfswork/rech/ads/uxp55sd/DP-SCR_Identify-and-estimate-density-lynx"
                                                "-population/data_pipeline/segment_anything/sam_vit_h_4b8939.pth") \
    .to(device="cuda")
predictor = SamPredictor(sam)

# Call the model on each image in our dataset
for idx in (pbar := tqdm(range(len(lynxDataset)))):
    content = lynxDataset[idx][0]
    image = content['image']
    conf = content['conf']
    x = content['x']
    y = content['y']
    width = content['width']
    height = content['height']
    filepath = content['filepath']
    filename = os.path.basename(filepath)
    pbar.set_postfix(image_shape=image.shape, conf=conf, filename=filename)

    # The bbox from MegaDetector provides more precise segmentation
    input_box = np.array([x, y, x + width, y + height])

    if not np.isnan(input_box).all():  # some images have no bounding box
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,  # we only want the segmentation with the highest score
        )
        mask = masks[0]

        # Get the segmented image
        image_masque = image.copy()
        image_masque[~mask, :] = 0
        image_masque = image_masque[int(y):int(y) + int(height), int(x):int(x) + int(width), :]

        # Save the segmented image
        image_masque_pil = Image.fromarray(image_masque)
        filepath_no_bg = f'{args.save_img_directory}/no_bg_{filename}'
        image_masque_pil.save(filepath_no_bg)

        all_filepath_no_bg.append(filepath_no_bg)

    else:
        all_filepath_no_bg.append(np.nan)

all_filepath_no_bg += [np.nan] * (len(csv) - len(all_filepath_no_bg))
csv['filepath_no_bg'] = all_filepath_no_bg
csv.to_csv("/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_full_no_bg.csv", index=False)

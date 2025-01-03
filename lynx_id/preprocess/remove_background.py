import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from lynx_id.data.dataset import LynxDataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="/gpfsscratch/rech/ads/commun/datasets/extracted"
                                                        "/lynx_dataset_full.csv")
    parser.add_argument('--save_img_directory', type=str, default="/gpfsscratch/rech/ads/commun/datasets/extracted"
                                                                  "/no_background")
    parser.add_argument('--skip_already_computed', default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def load_sam_model():
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint="/lustre/fswork/projects/rech/ads/commun/segment_anything/sam_vit_h_4b8939.pth") \
        .to(device="cuda")
    return SamPredictor(sam)


def remove_bg():
    args = parse_arguments()
    print(args.csv_file)

    csv = pd.read_csv(args.csv_file)
    # All paths to images without backgrounds to add them to the csv
    all_filepath_no_bg = []
    all_scores_sam = []

    # Load the dataset from csv
    lynxDataset = LynxDataset(Path(args.csv_file), probabilities=[1, 0, 0])

    # Load and init Segment Anything Model (SAM) on GPU
    predictor = load_sam_model()

    # Call the model on each image in our dataset
    for idx in (pbar := tqdm(range(len(lynxDataset)))):

        if args.skip_already_computed and 'filepath_no_bg' in csv.columns:
            filepath = csv.iloc[idx]['filepath_no_bg']
            if not pd.isna(filepath) and os.path.exists(filepath):
                all_filepath_no_bg.append(filepath)
                continue

        content = lynxDataset[idx]
        image = content[0]['image']
        conf = content[0]['conf']
        x = content[0]['x']
        y = content[0]['y']
        width = content[0]['width']
        height = content[0]['height']
        filepath = content[0]['filepath']
        filename = os.path.basename(filepath)
        image_number = content[0]['image_number']
        country = content[0]['country']
        lynx_id = content[1]['lynx_id']

        # The bbox from MegaDetector provides more precise segmentation
        input_box = np.array([x, y, x + width, y + height])

        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,  # we only want the segmentation with the highest score
        )
        mask = masks[0]

        # Get the segmented image
        image_mask = image.copy()
        image_mask[~mask, :] = 0
        image_mask = image_mask[int(y):int(y) + int(height), int(x):int(x) + int(width), :]

        # Save the segmented image
        image_masque_pil = Image.fromarray(image_mask)
        filepath_no_bg = f'{args.save_img_directory}/{country}/{lynx_id}/no_bg_{image_number}_{filename}'
        # {image_number} is particularly useful for Germany (several images have the same name but absolute filepath different)
        if os.path.exists(filepath_no_bg):
            print(f"Error: {filepath_no_bg}")

        if not os.path.exists(os.path.dirname(filepath_no_bg)):
            os.makedirs(os.path.dirname(filepath_no_bg))
        image_masque_pil.save(filepath_no_bg)

        all_filepath_no_bg.append(filepath_no_bg)
        all_scores_sam.append(scores[0])

        pbar.set_postfix(image_shape=image.shape, conf=conf, score_sam=scores[0], filename=filename)

    csv['filepath_no_bg'] = all_filepath_no_bg
    csv['score_sam'] = all_scores_sam
    csv.to_csv(args.csv_file, index=False)

    return csv


if __name__ == "__main__":
    remove_bg()

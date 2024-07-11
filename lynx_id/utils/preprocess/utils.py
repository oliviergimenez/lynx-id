import os
import time
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lynx_id.utils.megadetector.utils import crop_bbox


def remove_basename_duplicates(df, keep_first=True):
    df['basename'] = df['filepath'].apply(os.path.basename)
    if keep_first:
        return df.drop_duplicates(subset='basename').drop(columns=['basename'])
    else:  # remove all samples
        duplicated_basenames = df['basename'][df['basename'].duplicated(keep=False)]
        df = df[~df['basename'].isin(duplicated_basenames)]
        return df.drop(columns=['basename'])


def segmentation_inversion(row_df, save_new_img=False):
    base_img = crop_bbox(row_df)
    segmented_img = Image.open(row_df['filepath_no_bg'])
    segmented_img = segmented_img.resize(base_img.size)

    width, height = base_img.size

    inverse_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel_base = base_img.getpixel((x, y))
            pixel_segmented = segmented_img.getpixel((x, y))

            if pixel_segmented == (0, 0, 0):  # noir opaque
                inverse_image.putpixel((x, y), pixel_base)
            else:
                inverse_image.putpixel((x, y), (0, 0, 0))

    if save_new_img:
        inverse_image.save(row_df['filepath_no_bg'])

    return inverse_image, base_img


def check_filepath(NO_BACKGROUND, COUNTRY, filepath, lynx_id, image_number):
    filepath_no_bg = NO_BACKGROUND / COUNTRY / lynx_id / Path(f"no_bg_{image_number}_{os.path.basename(filepath)}")
    if os.path.exists(filepath_no_bg):
        return filepath_no_bg
    else:
        return np.nan


def get_no_and_multiple_bbox(bbox_dict):
    no_bbox = []
    multiple_bbox = []

    for img in bbox_dict['images']:
        if not 'detections' in img:
            no_bbox.append(img['file'])
        elif not img['detections']:
            no_bbox.append(img['file'])
        elif len(img['detections']) > 1:
            multiple_bbox.append(img)

    print(f"{len(no_bbox)} images have no bounding boxes detected.")
    print(f"{len(multiple_bbox)} images have several bounding boxes detected.")
    print(f"Total: {len(bbox_dict['images'])} images.")

    return no_bbox, multiple_bbox


def flatten_bbox(bbox_dict, add_image_without_bbox=True, verbose=False):
    flat_data = []
    for img in bbox_dict['images']:
        if 'detections' in img and img['detections']:
            for detection in img['detections']:
                flat_data.append({'file': img['file'],
                                  'im_width': img['width'],
                                  'im_height': img['height'],
                                  'category': detection['category'],
                                  'conf': detection['conf'],
                                  'x': detection['bbox'][0],
                                  'y': detection['bbox'][1],
                                  'width': detection['bbox'][2],
                                  'height': detection['bbox'][3]})
        else:
            if verbose:
                print(f"No bbox in {img['file']}")
            if add_image_without_bbox:
                flat_data.append({'file': img['file'],
                                  'im_width': img['width'],
                                  'im_height': img['height'],
                                  'category': np.nan,
                                  'conf': np.nan,
                                  'x': np.nan,
                                  'y': np.nan,
                                  'width': np.nan,
                                  'height': np.nan})

    return pd.DataFrame(flat_data)


def absolute_coordinates_bbox(df_bbox):
    # Follow the "coco" convention (x_min, y_min, width, height)

    df_bbox['x'] *= df_bbox['im_width']
    df_bbox['y'] *= df_bbox['im_height']
    df_bbox['width'] *= df_bbox['im_width']
    df_bbox['height'] *= df_bbox['im_height']

    return df_bbox


def separate_single_multiple_df(df_bbox):
    bbox_counts = df_bbox['file'].value_counts()

    df_bbox_single_detection = df_bbox[
        df_bbox['file'].isin(bbox_counts[bbox_counts == 1].index)]
    df_bbox_multiple_detections = df_bbox[
        df_bbox['file'].isin(bbox_counts[bbox_counts > 1].index)]

    df_bbox_single_detection.reset_index(drop=True, inplace=True)
    df_bbox_multiple_detections.reset_index(drop=True, inplace=True)

    return df_bbox_single_detection, df_bbox_multiple_detections


def plot_images_conf(df_bbox, by='largest'):
    if by == 'largest':
        by_conf = df_bbox.nlargest(10, 'conf')
    else:
        by_conf = df_bbox.nsmallest(10, 'conf')

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (file_idx, ax) in enumerate(zip(by_conf.index, axes)):
        ax.imshow(crop_bbox(df_bbox.iloc[file_idx]))
        ax.set_title(df_bbox.iloc[file_idx]['file'].split("/")[-1] + " + conf : " +
                     str(df_bbox.iloc[file_idx]['conf']))
        ax.axis('off')

    plt.show()


def measure_performance(dataset, num_samples=100):
    start_time = time.time()
    for i in range(num_samples):
        _ = dataset[i]
    end_time = time.time()
    return end_time - start_time



def csv_random_equilibrated_splitter(csv_path):
    
    return 0
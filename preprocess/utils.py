import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../")

from data_pipeline.megadetector.utils import crop_bbox


def norm_bbox():
    print('done')

def get_no_and_multiple_bbox(bbox_dict):
    no_bbox = []
    multiple_bbox = []

    for img in bbox_dict['images']:
        if not img['detections']:
            no_bbox.append(img['file'])
        elif len(img['detections']) > 1:
            multiple_bbox.append(img)

    print(f"{len(no_bbox)} images have no bounding boxes detected.")
    print(f"{len(multiple_bbox)} images have several bounding boxes detected.")
    print(f"Total: {len(bbox_dict['images'])} images.")

    return no_bbox, multiple_bbox


def flatten_bbox(bbox_dict, add_image_without_bbox=True):
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

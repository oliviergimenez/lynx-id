from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LynxDataset(Dataset):
    def __init__(self, dataset_csv: Path, loader='pil', transform=None,
                 augmentation=None):
        self.dataset_csv = dataset_csv
        self.dataframe = pd.read_csv(dataset_csv)
        self.loader = loader  # 'pil' or 'opencv'
        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]

        # Load the image using the specified loader
        if self.loader == 'opencv':
            img = cv2.imread(image_id["filepath"])
            if img is None:
                raise ValueError(
                    f"Image not found or corrupted at {image_id['filepath']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.loader == 'pil':
            img = Image.open(image_id["filepath"])
            img = np.array(img.convert('RGB'))  # Convert to RGB
        else:
            raise ValueError("Unsupported loader. Choose 'pil' or 'opencv'.")

        # Apply transformations (e.g., resizing)
        if self.transform:
            img = self.transform(image=img)['image']

        # Apply augmentations
        if self.augmentation:
            img = self.augmentation(image=img)['image']

        # Prepare the input and output dictionaries
        input_dict = {
            'image': img,
            'source': image_id["source"],
            'pattern': image_id["pattern"],
            'date': image_id["date"],
            'location': image_id["location"],
            'image_number': image_id["image_number"],
            'conf': image_id["conf"],
            'x': image_id["x"],
            'y': image_id["y"],
            'width': image_id["width"],
            'height': image_id["height"],
            'filepath': image_id["filepath"]
        }

        output_dict = {
            'lynx_id': image_id["lynx_id"]
        }

        return input_dict, output_dict

    def __len__(self):
        return len(self.dataframe)


def collate(batch):
    # Initialize lists to gather all elements for each key
    images = []
    sources = []
    patterns = []
    dates = []
    locations = []
    image_numbers = []
    lynx_ids = []

    # Iterate over each item in the batch
    for input_dict, output_dict in batch:
        # Append data from input dictionary
        images.append(input_dict['image'])  # List of images
        sources.append(input_dict['source'])
        patterns.append(input_dict['pattern'])
        dates.append(input_dict['date'])
        locations.append(input_dict['location'])
        image_numbers.append(input_dict['image_number'])

        # Append data from output dictionary
        lynx_ids.append(output_dict['lynx_id'])

    # Construct the batched input and output dictionaries
    batched_input_dict = {
        'images': images,
        # conversion to array not possible because as image size varies
        'sources': sources,
        'patterns': patterns,
        'dates': dates,
        'locations': locations,
        'image_numbers': image_numbers
    }

    batched_output_dict = {
        'lynx_ids': lynx_ids
    }

    return batched_input_dict, batched_output_dict

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LynxDataset(Dataset):
    def __init__(self, dataset_csv: Path, loader='pil', transform=None,
                 augmentation=None, probabilities=[1/3, 1/3, 1/3]):
        self.dataset_csv = dataset_csv
        self.dataframe = pd.read_csv(dataset_csv)
        self.loader = loader  # 'pil' or 'opencv'
        self.transform = transform
        self.augmentation = augmentation

        # Type of image to load (classic, bounding box, no background) with a given probability
        self.image_types = ["classic", "bbox", "no_bg"]
        self.probabilities = probabilities

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]
        
        image_type_choice = random.choices(self.image_types, weights=self.probabilities)[0]

        filepath = image_id["filepath"] if image_type_choice != "no_bg" else image_id["filepath_no_bg"]

        # Load the image using the specified loader
        if self.loader == 'opencv':
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError(
                    f"Image not found or corrupted at {image_id['filepath']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.loader == 'pil':
            img = Image.open(filepath)
            img = np.array(img.convert('RGB'))  # Convert to RGB
        else:
            raise ValueError("Unsupported loader. Choose 'pil' or 'opencv'.")

        # Apply transformations (e.g.: resizing, bounding box...)
        if image_type_choice == "bbox":
            bbox = image_id[['x', 'y', 'width', 'height']].values.tolist()
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0]+bbox[2]
            y_max = bbox[1]+bbox[3]

            img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

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
            'filepath': filepath
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

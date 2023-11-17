from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LynxDataset(Dataset):
    def __init__(self, dataset_csv: Path, loader='pil'):
        self.dataset_csv = dataset_csv
        self.dataframe = pd.read_csv(dataset_csv)
        self.loader = loader  # 'pil' or 'opencv'
                
    def __getitem__(self, idx):
        # Retrieve the row corresponding to the index
        image_id = self.dataframe.iloc[idx]

        # Load the image using the specified loader
        if self.loader == 'opencv':
            img = cv2.imread(image_id["filepath"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise ValueError(f"Image not found or corrupted at {image_id['filepath']}")
        elif self.loader == 'pil':
            img = Image.open(image_id["filepath"])
            img = np.array(img.convert('RGB'))  # Convert to RGB
        else:
            raise ValueError("Unsupported loader. Choose 'pil' or 'opencv'.")

        # Prepare the input dictionary with the image and other data
        input_dict = {
            'image': img,
            'source': image_id["source"],
            'pattern': image_id["pattern"],
            'date': image_id["date"],
            'location': image_id["location"],
            'image_number': image_id["image_number"]
        }

        # Prepare the output dictionary with lynx_id
        output_dict = {
            'lynx_id': image_id["lynx_id"]
        }

        return input_dict, output_dict
        
    def __len__(self):
        return len(self.dataframe)

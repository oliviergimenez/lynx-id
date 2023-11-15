from torch.utils.data import Dataset
import cv2
from pathlib import Path

import pandas as pd

class LynxDataset(Dataset):
    def __init__(self,
                 dataset_csv: Path
    ):
        self.dataset_csv = dataset_csv
        self.dataframe = pd.read_csv(dataset_csv)
                
    def __getitem__(self, idx):
        # Retrieve the row corresponding to the index
        image_id = self.dataframe.iloc[idx]

        # Load the image
        img = cv2.imread(image_id["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        
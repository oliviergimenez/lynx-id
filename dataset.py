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
        image_id = self.dataframe.iloc[idx]
        
        img = cv2.imread(image_id["filepath"])
        
        return img
        
    def __len__(self):
        return len(self.dataframe)
        

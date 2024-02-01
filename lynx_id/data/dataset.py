import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_single

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LynxDataset(Dataset):
    def __init__(self, dataset_csv: Path, loader='pil', transform=None, augmentation=None, mode='single',
                 probabilities=[1 / 3, 1 / 3, 1 / 3], load_triplet_path=None, save_triplet_path=None, model=None,
                 device='auto', verbose=False):
        self.dataset_csv = dataset_csv
        self.dataframe = pd.read_csv(dataset_csv)
        self.has_filepath_no_bg = True if "filepath_no_bg" in self.dataframe.columns else False
        self.loader = loader
        self.transform = transform
        self.augmentation = augmentation
        self.mode = mode
        # Type of image to load (classic, bounding box, no background) with a given probability
        self.image_types = ["classic", "bbox", "no_bg"]
        self.probabilities = probabilities
        self.load_triplet_path = load_triplet_path
        self.save_triplet_path = save_triplet_path
        self.model = model
        self.verbose = verbose

        self.sampling_strategy = "random"

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.mode == 'triplet':
            if self.load_triplet_path and os.path.exists(self.load_triplet_path):
                self.load_triplet_precompute()
            else:
                if self.model is None:
                    raise ValueError("A model must be provided for 'triplet' mode.")
                self.compute_embeddings_and_distances()
                if self.save_triplet_path:
                    self.save_triplet_precompute()

    def save_triplet_precompute(self):
        # Convert PyTorch tensors to NumPy arrays before saving
        embeddings_np = self.embeddings.cpu().numpy()
        distance_matrix_np = self.distance_matrix.cpu().numpy()
        # lynx_ids = self.lynx_ids.numpy()
        np.savez(self.save_triplet_path, embeddings=embeddings_np, distance_matrix=distance_matrix_np,
                 lynx_ids=self.lynx_ids)

    def load_triplet_precompute(self):
        try:
            data = np.load(self.load_triplet_path)
            self.embeddings = torch.tensor(data['embeddings'])  # Defaults to CPU
            self.distance_matrix = torch.tensor(data['distance_matrix'])  # Defaults to CPU
            self.lynx_ids = list(data['lynx_ids'])
        except IOError:
            print(f"Error loading file: {self.load_triplet_path}. Check if the file exists and is not corrupted.")

    def compute_embeddings_and_distances(self):
        # Ensure model is on the right device
        self.model = self.model.to(self.device)
        # Ensure model is in evaluation mode
        self.model.eval()

        # Temporarily switch to 'single' mode for embedding computation
        original_mode = self.mode
        self.mode = 'single'

        # DataLoader with the custom collate function
        # Consider setting these values based on your system's capabilities
        # ADD AUTO BATCH, auto_numwork
        loader = DataLoader(self, batch_size=64, shuffle=False, num_workers=10, prefetch_factor=2,
                            collate_fn=collate_single)

        # List to store embeddings and lynx IDs
        all_embeddings = []
        all_lynx_ids = []

        # Iterate over the dataset using DataLoader        
        for batched_input_dict, batched_output_dict in tqdm(loader, desc="Processing images", disable=not self.verbose):
            # Access the batched images and lynx IDs
            batch_images = torch.stack(batched_input_dict['image']).to(
                self.device)  # Ensure data is on the same device as model
            batch_lynx_ids = batched_output_dict['lynx_id']

            # Compute embeddings
            with torch.no_grad():
                embeddings = self.model(batch_images)
                embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings.cpu())  # Move embeddings to CPU to conserve GPU memory

            # Collect lynx IDs
            all_lynx_ids.extend(batch_lynx_ids)

        # Concatenate all embeddings
        self.embeddings = torch.cat(all_embeddings, dim=0)
        # Compute the distance matrix
        self.distance_matrix = torch.cdist(self.embeddings, self.embeddings, p=2)
        self.lynx_ids = all_lynx_ids

        # Revert to the original mode
        self.mode = original_mode

    def load_image(self, filepath):
        # Ensure filepath is a string
        filepath = str(filepath)

        # Load the image using the specified loader
        if self.loader == 'opencv':
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError(f"Image not found or corrupted at {filepath}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.loader == 'pil':
            img = Image.open(filepath)
            img = np.array(img.convert('RGB'))  # Convert to RGB
        else:
            raise ValueError("Unsupported loader. Choose 'pil' or 'opencv'.")
        return img

    def apply_transforms(self, img):
        # Apply transformations (e.g., resizing)
        if self.transform:
            img = self.transform(image=img)['image']
        # Apply augmentations
        if self.augmentation:
            img = self.augmentation(image=img)['image']
        return img

    def prepare_data(self, info):
        if self.has_filepath_no_bg:
            image_type_choice = random.choices(self.image_types, weights=self.probabilities)[0]
            filepath = info["filepath"] if image_type_choice != "no_bg" else info["filepath_no_bg"]
        else:
            image_type_choice = self.image_types[0]  # classic image
            filepath = info["filepath"]

        img = self.load_image(filepath)

        if image_type_choice == "bbox":
            bbox = info[['x', 'y', 'width', 'height']].values.tolist()
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]

            img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

        img = self.apply_transforms(img)

        # Prepare the input and output dictionaries
        input_dict = {
            'image': img,
            'source': info["source"],
            'pattern': info["pattern"],
            'date': info["date"],
            'location': info["location"],
            'image_number': info["image_number"],
        }

        output_dict = {
            'lynx_id': info["lynx_id"]
        }

        return input_dict, output_dict

    def get_single_item(self, idx):
        image_info = self.dataframe.iloc[idx]
        input_dict, output_dict = self.prepare_data(image_info)
        return input_dict, output_dict

    def get_triplet_item_old(self, idx):
        anchor_info = self.dataframe.iloc[idx]
        anchor_input, anchor_output = self.prepare_data(anchor_info)

        # Corrected: Randomly select a positive sample
        positive_indices = [i for i in range(len(self.dataframe)) if
                            self.dataframe.iloc[i]['lynx_id'] == anchor_info['lynx_id'] and i != idx]
        positive_idx = random.choice(positive_indices) if positive_indices else idx
        positive_info = self.dataframe.iloc[positive_idx]
        positive_input, positive_output = self.prepare_data(positive_info)

        # Corrected: Randomly select a negative sample
        negative_indices = [i for i in range(len(self.dataframe)) if
                            self.dataframe.iloc[i]['lynx_id'] != anchor_info['lynx_id']]
        negative_idx = random.choice(negative_indices)
        negative_info = self.dataframe.iloc[negative_idx]
        negative_input, negative_output = self.prepare_data(negative_info)

        # Prepare nested dictionaries for anchor, positive, and negative
        data = {
            'anchor': {
                'input': anchor_input,
                'output': anchor_output
            },
            'positive': {
                'input': positive_input,
                'output': positive_output
            },
            'negative': {
                'input': negative_input,
                'output': negative_output
            }
        }

        return data

    def get_triplet_item_old(self, idx):
        anchor_info = self.dataframe.iloc[idx]
        anchor_input, anchor_output = self.prepare_data(anchor_info)

        # Implement different sampling strategies
        if self.sampling_strategy == 'random':
            positive_idx, negative_idx = self.random_sampling(anchor_info)
        if self.sampling_strategy == 'hard':
            # Assuming positive sampling remains random
            positive_idx = self.random_sampling(anchor_info, idx)[0]
            negative_idx = self.hard_sampling(anchor_info, idx)

        positive_info = self.dataframe.iloc[positive_idx]
        positive_input, positive_output = self.prepare_data(positive_info)

        negative_info = self.dataframe.iloc[negative_idx]
        negative_input, negative_output = self.prepare_data(negative_info)

        data = {
            'anchor': {'input': anchor_input, 'output': anchor_output},
            'positive': {'input': positive_input, 'output': positive_output},
            'negative': {'input': negative_input, 'output': negative_output}
        }

        return data

    def get_triplet_item(self, idx):
        if self.sampling_strategy == 'random':
            data = self.random_sampling(idx)
        elif self.sampling_strategy == 'hard':
            data = self.hard_sampling(idx)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        return data

    def random_sampling(self, anchor_idx):
        # Load anchor
        anchor_info = self.dataframe.iloc[anchor_idx]
        anchor_input, anchor_output = self.prepare_data(anchor_info)

        # Randomly select a positive sample
        positive_indices = [i for i in range(len(self.dataframe)) if
                            self.dataframe.iloc[i]['lynx_id'] == anchor_info['lynx_id'] and i != anchor_idx]
        positive_idx = random.choice(positive_indices) if positive_indices else anchor_idx
        positive_info = self.dataframe.iloc[positive_idx]
        positive_input, positive_output = self.prepare_data(positive_info)

        # Randomly select a negative sample
        negative_indices = [i for i in range(len(self.dataframe)) if
                            self.dataframe.iloc[i]['lynx_id'] != anchor_info['lynx_id']]
        negative_idx = random.choice(negative_indices)
        negative_info = self.dataframe.iloc[negative_idx]
        negative_input, negative_output = self.prepare_data(negative_info)

        if self.verbose:
            # Compute distances only if verbose mode is on
            anchor_embedding = self.embeddings[anchor_idx]
            distances = torch.norm(self.embeddings - anchor_embedding, dim=1)
            positive_distance = distances[positive_idx].item()
            negative_distance = distances[negative_idx].item()
            print(f"Random positive distance for anchor {anchor_idx}: {positive_distance}")
            print(f"Random negative distance for anchor {anchor_idx}: {negative_distance}")

        data = {
            'anchor': {'input': anchor_input, 'output': anchor_output},
            'positive': {'input': positive_input, 'output': positive_output},
            'negative': {'input': negative_input, 'output': negative_output}
        }
        return data

    def hard_sampling(self, anchor_idx):
        # Load anchor
        anchor_info = self.dataframe.iloc[anchor_idx]
        anchor_input, anchor_output = self.prepare_data(anchor_info)

        # Precomputed embeddings and lynx IDs should be available
        anchor_embedding = self.embeddings[anchor_idx]
        distances = torch.norm(self.embeddings - anchor_embedding, dim=1)  # L1 distance
        distances[anchor_idx] = float('inf')  # Ignore the anchor itself

        # Assuming positive sampling remains random
        positive_indices = [i for i in range(len(self.dataframe)) if
                            self.dataframe.iloc[i]['lynx_id'] == anchor_info['lynx_id'] and i != anchor_idx]
        # hard_positive_idx = positive_indices[torch.argmax(positive_distances).item()]
        positive_idx = random.choice(positive_indices) if positive_indices else anchor_idx
        positive_info = self.dataframe.iloc[positive_idx]
        positive_input, positive_output = self.prepare_data(positive_info)

        # Find the hard negative
        negatives = [i for i, lynx_id in enumerate(self.lynx_ids) if lynx_id != anchor_info['lynx_id']]
        negative_distances = distances[negatives]
        hard_negative_idx = negatives[torch.argmin(negative_distances).item()]
        hard_negative_info = self.dataframe.iloc[hard_negative_idx]
        hard_negative_input, hard_negative_output = self.prepare_data(hard_negative_info)

        # Debugging: Print the distance of the hard negative if verbose mode is on
        if self.verbose:
            hard_negative_distance = negative_distances.min().item()
            print(f"Hard negative distance for anchor {anchor_idx}: {hard_negative_distance}")

        data = {
            'anchor': {'input': anchor_input, 'output': anchor_output},
            'positive': {'input': positive_input, 'output': positive_output},
            'negative': {'input': hard_negative_input, 'output': hard_negative_output}
        }

        return data

    def __getitem__(self, idx):
        if self.mode == 'single':
            return self.get_single_item(idx)
        elif self.mode == 'triplet':
            return self.get_triplet_item(idx)
        else:
            raise ValueError("Invalid mode. Choose 'single' or 'triplet'.")

    def __len__(self):
        return len(self.dataframe)

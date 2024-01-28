from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Sampler, Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


from collections import defaultdict
import numpy as np
import random

from PIL import Image
import cv2
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class LynxDataset(Dataset):
    def __init__(self, dataset_csv: Path, loader='pil', transform=None, augmentation=None, mode='single', load_triplet_path=None,
                 save_triplet_path=None, model=None, device='auto', verbose=False):
        self.dataset_csv = dataset_csv
        self.dataframe = pd.read_csv(dataset_csv)
        self.loader = loader
        self.transform = transform
        self.augmentation = augmentation
        self.mode = mode
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
        #lynx_ids = self.lynx_ids.numpy()
        np.savez(self.save_triplet_path, embeddings=embeddings_np, distance_matrix=distance_matrix_np, lynx_ids=self.lynx_ids)

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
        loader = DataLoader(self, batch_size=64, shuffle=False, num_workers=10, prefetch_factor=2, collate_fn=collate_single)

        # List to store embeddings and lynx IDs
        all_embeddings = []
        all_lynx_ids = []
        
        # Iterate over the dataset using DataLoader        
        for batched_input_dict, batched_output_dict in tqdm(loader, desc="Processing images", disable=not self.verbose):
            # Access the batched images and lynx IDs
            batch_images = torch.stack(batched_input_dict['image']).to(self.device)  # Ensure data is on the same device as model
            batch_lynx_ids = batched_output_dict['lynx_id']
            
            # Compute embeddings
            with torch.no_grad():
                embeddings = self.model(batch_images)
                embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings.cpu()) # Move embeddings to CPU to conserve GPU memory
            
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
        img = self.load_image(info["filepath"])
        img = self.apply_transforms(img)

        # Prepare the input and output dictionaries
        input_dict = {
            'image': img,
            'source': info["source"],
            'pattern': info["pattern"],
            'date': info["date"],
            'location': info["location"],
            'image_number': info["image_number"]
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
        positive_indices = [i for i in range(len(self.dataframe)) if self.dataframe.iloc[i]['lynx_id'] == anchor_info['lynx_id'] and i != idx]
        positive_idx = random.choice(positive_indices) if positive_indices else idx
        positive_info = self.dataframe.iloc[positive_idx]
        positive_input, positive_output = self.prepare_data(positive_info)

        # Corrected: Randomly select a negative sample
        negative_indices = [i for i in range(len(self.dataframe)) if self.dataframe.iloc[i]['lynx_id'] != anchor_info['lynx_id']]
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
        positive_indices = [i for i in range(len(self.dataframe)) if self.dataframe.iloc[i]['lynx_id'] == anchor_info['lynx_id'] and i != anchor_idx]
        positive_idx = random.choice(positive_indices) if positive_indices else anchor_idx
        positive_info = self.dataframe.iloc[positive_idx]
        positive_input, positive_output = self.prepare_data(positive_info)

        # Randomly select a negative sample
        negative_indices = [i for i in range(len(self.dataframe)) if self.dataframe.iloc[i]['lynx_id'] != anchor_info['lynx_id']]
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
        distances = torch.norm(self.embeddings - anchor_embedding, dim=1) # L1 distance
        distances[anchor_idx] = float('inf')  # Ignore the anchor itself

        # Assuming positive sampling remains random
        positive_indices = [i for i in range(len(self.dataframe)) if self.dataframe.iloc[i]['lynx_id'] == anchor_info['lynx_id'] and i != anchor_idx]
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



class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.class_counts = defaultdict(int)

        # Count instances per class
        for _, output in dataset:
            lynx_id = output['lynx_id']
            self.class_counts[lynx_id] += 1

        # Calculate weights for each instance
        self.weights = [1.0 / self.class_counts[dataset[idx][1]['lynx_id']] for idx in self.indices]

    def __iter__(self):
        # Sample anchors based on weights
        anchors = torch.multinomial(torch.tensor(self.weights, dtype=torch.double), len(self.weights), replacement=True)
        return (self.indices[i] for i in anchors)

    def __len__(self):
        return len(self.indices)


def collate_single(batch):
    if not batch:
        return {}, {}

    # Sample the first item to get the keys
    first_input, first_output = batch[0]
    batched_input_dict = {key: [] for key in first_input.keys()}
    batched_output_dict = {key: [] for key in first_output.keys()}
    
    # Iterate over each item in the batch
    for input_dict, output_dict in batch:
        # Append data from input and output dictionaries
        for key in input_dict:
            batched_input_dict[key].append(input_dict[key])
        for key in output_dict:
            batched_output_dict[key].append(output_dict[key])

    return batched_input_dict, batched_output_dict    
    
def collate(batch):
    #Old style, to be removed
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


def collate_triplet_old(batch):
    if not batch:
        return {}

    # Initialize nested dictionaries for the batch
    batched_data = {
        'anchor': {'input': [], 'output': []},
        'positive': {'input': [], 'output': []},
        'negative': {'input': [], 'output': []}
    }

    # Iterate over each triplet in the batch
    for triplet in batch:
        for key in ['anchor', 'positive', 'negative']:
            for subkey in ['input', 'output']:
                batched_data[key][subkey].append(triplet[key][subkey])

    return batched_data


def collate_triplet(batch):
    if not batch:
        return {}

    # Initialize nested dictionaries for the batch
    batched_data = {
        'anchor': {'input': {}, 'output': {}},
        'positive': {'input': {}, 'output': {}},
        'negative': {'input': {}, 'output': {}}
    }
    
    # Iterate over each triplet in the batch
    for triplet in batch:
        for key in ['anchor', 'positive', 'negative']:
            for subkey in ['input', 'output']:
                for feature_key, feature_value in triplet[key][subkey].items():
                    if feature_key not in batched_data[key][subkey]:
                        batched_data[key][subkey][feature_key] = []
                    batched_data[key][subkey][feature_key].append(feature_value)
                
                
    # Post-process features if necessary (e.g., stacking 'image' tensors)
    for key in ['anchor', 'positive', 'negative']:
        if 'image' in batched_data[key]['input']:
            images = batched_data[key]['input']['image']
            batched_data[key]['input']['image'] = torch.stack(images)
        # Additional post-processing for other features can be added here
    return batched_data

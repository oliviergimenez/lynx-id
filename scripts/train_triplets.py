import os
import sys
import argparse
# Construct project root path using the WORK environment variable
work_dir = os.environ.get('WORK')
folder_name = 'DP-SCR_Identify-and-estimate-density-lynx-population'
project_root = os.path.join(work_dir, folder_name)
sys.path.append(project_root)

# Importing LynxDataset class and related elements
from data_pipeline.triplets import LynxDataset
from data_pipeline.triplets import collate_triplet
from data_pipeline.transformations_and_augmentations import transforms, augments

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.models import resnet50 
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


#from tqdm.notebook import tqdm
from tqdm import tqdm
import time


def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model on lynx dataset using triplet loss.')
    parser.add_argument('--csv', type=str, required=False, help='Path to dataset CSV file', default='/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_full.csv')
    parser.add_argument('--model_weights', type=str, required=False, help='Path to the pretrained model', default='/gpfsscratch/rech/ads/commun/models/resnet50/pretrained_weights.pt')
    parser.add_argument('--save_path', type=str, default='triplet_precompute', help='Path to save precomputed triplets')
    parser.add_argument('--load_path', type=str, help='Path to load precomputed triplets', default='/gpfsscratch/rech/ads/commun/precompute/triplet_precompute.npz')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], help='Device to use for training', default='cuda')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    return args

def main(args):  
    # Path for saving models and TensorBoard logs
    experiment_name = "kg_tests"
    experiment_path = os.path.join(os.environ.get('ALL_CCFRWORK'), experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(experiment_path)

    # Initialize dataset
    dataset = LynxDataset(dataset_csv=args.csv, 
                          loader="pil",
                          transform=transforms,  # Define 'preprocess' earlier in your script
                          augmentation=None,
                          mode='triplet',
                          load_triplet_path=args.load_path,
                          save_triplet_path=args.save_path,
                          model=torch.load(args.model_weights),
                          device=args.device, 
                          verbose=args.verbose)
    
    dataloader = DataLoader(dataset, 
                            batch_size=64, 
                            shuffle=True, 
                            collate_fn=collate_triplet,
                            prefetch_factor=8,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True)
    
    
    # Initialize model
    model_weights = torch.load(args.model_weights)
    model = models.resnet50(pretrained=False)
    model.load_state_dict(model_weights)        
    model.fc = nn.Identity()  # Replace the final fully connected layer
    model.to(args.device)

    
    # Triplet Loss
    triplet_loss = nn.TripletMarginLoss(margin=1.0)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    num_epochs = 10  # Example epoch count
    # Assuming 'optimizer' is already defined
    T_max = num_epochs  # Here, we set it to the total number of epochs for one cycle
    eta_min = 0.0001  # The minimum learning rate, adjust as needed

    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    # Path for saving models
    save_path = '/gpfswork/rech/ads/commun/models/triplet_embeddings'
    os.makedirs(save_path, exist_ok=True)

    best_loss = float('inf')

    
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(enumerate(dataloader), miniters=1, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as dataloader_tqdm:
            for i, batch in dataloader_tqdm:
                optimizer.zero_grad()

                # Assign dictionaries to variables
                anchor = batch['anchor']
                positive = batch['positive']
                negative = batch['negative']

                # Move images to the correct device directly
                anchor['input']['image'] = anchor['input']['image'].to(args.device).float()
                positive['input']['image'] = positive['input']['image'].to(args.device).float()
                negative['input']['image'] = negative['input']['image'].to(args.device).float()
                
                with autocast():
                    # Forward pass using the directly accessed images
                    # Maybe it is better to forward all at the same time but it will reduce batch size by 3 
                    anchor_embedding = model(anchor['input']['image'])
                    positive_embedding = model(positive['input']['image'])
                    negative_embedding = model(negative['input']['image'])

                    # Compute triplet loss
                    loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
                    
                    
                # Backward pass and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update tqdm bar with current loss
                dataloader_tqdm.set_postfix(loss=loss.item())

                # Log loss for each batch
                writer.add_scalar('Loss/Batch', loss.item(), epoch * len(dataloader) + i)
                epoch_loss += loss.item()


        # Average loss for the epoch
        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
               
        # Save the last model
        last_model_path = os.path.join(experiment_path, f'model_last_{epoch_loss:.3f}.pth')
        torch.save(model.state_dict(), last_model_path)

        # Check if this is the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(experiment_path, f'model_best_{best_loss:.3f}.pth')
            torch.save(model.state_dict(), best_model_path)

    # Close the TensorBoard writer
    writer.close()

    print(f"Best model saved at: {best_model_path}")
    print(f"Last model saved at: {last_model_path}")    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Script end")

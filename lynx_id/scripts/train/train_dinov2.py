# lynx_id/scripts/train/train_triplets.py

import os
import datetime
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

# Assuming LynxDataset, collate_triplet, and other necessary modules are correctly defined
# in your package and the paths are correct.
from lynx_id.data.dataset import LynxDataset
from lynx_id.data.collate import collate_triplet, collate_single
from lynx_id.data.transformations_and_augmentations import transforms
from lynx_id.eval.eval import EvalMetrics
from lynx_id.model.clustering import ClusteringModel
from lynx_id.model.embeddings import EmbeddingModel
from lynx_id.utils import dinov2_utils




def create_parser():
    """Create and return the argument parser for the training triplets script."""
    parser = argparse.ArgumentParser(description='Train a model on lynx dataset using triplet loss.')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to CSV file for training dataset')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to CSV file for validation dataset')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to CSV file for test dataset')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save precomputed triplets')
    parser.add_argument('--load_path', type=str, required=True, help='Path to load precomputed triplets')
    parser.add_argument('--experiment_path', type=str, required=True, help='Path for saving models and logs')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='cuda',
                        help='Device to use for training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--debug', action='store_true', help='if enabled, iterates only on 10 batches')
    return parser


def create_dataloader(dataset, shuffle, collate_fn, batch_size=32, num_workers=4, pin_memory=True,
                      persistent_workers=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        prefetch_factor=8,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


def main(args):
    print("Running train_triplets with arguments:", args)

    # Initial setup
    # Path for saving models
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.experiment_path, exist_ok=True)

    # Initialize TensorBoard writer
    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')    
    writer = SummaryWriter(os.path.join(args.experiment_path, subdir))
    #writer = SummaryWriter(args.experiment_path)

    # Model embedder from triplet distance computation
    model_embedder_weights = torch.load(args.model_weights)
    model_embedder = models.resnet50(pretrained=False)
    model_embedder.load_state_dict(model_embedder_weights)

    
    
    # Dataset initialization
    train_dataset_triplet = LynxDataset(
        dataset_csv=args.train_csv,
        loader="pil",
        transform=transforms,  # Ensure 'transforms' is defined or imported correctly
        augmentation=None,
        probabilities=[0, 0.5, 0.5],        
        mode='triplet',
        load_triplet_path=args.load_path,
        save_triplet_path=args.save_path,
        model=model_embedder,
        device=args.device,
        verbose=args.verbose
    )

    # train dataset for evaluation (single mode)
    train_dataset_single = LynxDataset(
        dataset_csv=args.train_csv,
        loader="pil",
        transform=transforms,
        augmentation=None,
        probabilities=[0, 0, 1],
        mode='single',
        device=args.device
    )  # Mandatory, since triplet mode produces classic, bounding-box and backgroundless images.
    # For evaluation, we want no_bg images.

    val_dataset = LynxDataset(
        dataset_csv=args.val_csv,
        loader="pil",
        transform=transforms,
        augmentation=None,
        probabilities=[0, 0, 1],
        mode='single',
        device="auto"
    )  # useful for computing the threshold for detecting new individuals when evaluating the test set

    test_dataset = LynxDataset(
        dataset_csv=args.test_csv,
        loader="pil",
        transform=transforms,
        augmentation=None,
        probabilities=[0, 0, 1],
        mode='single',
        device="auto"
    )

    # Dataloader initialization
    train_dataloader_triplet = create_dataloader(dataset=train_dataset_triplet, shuffle=True,
                                                 collate_fn=collate_triplet)
    train_dataloader_single = create_dataloader(dataset=train_dataset_single, shuffle=False, collate_fn=collate_single)
    val_dataloader = create_dataloader(dataset=val_dataset, shuffle=False, collate_fn=collate_single)
    test_dataloader = create_dataloader(dataset=test_dataset, shuffle=False, collate_fn=collate_single)


    # Update dataset lynx_id lists by updating lynx_ids not present in the training set
    val_lynx_id = val_dataset.compute_new_lynx_id(train_dataset_single)
    test_lynx_id = test_dataset.compute_new_lynx_id(train_dataset_single)

    # Define top_k for evaluation (CMC@k and mAP@k metrics)
    top_k = (1, 2, 3, 4, 5)

    # Model initialization
    embedding_model = EmbeddingModel(
        model_path=args.model_weights,
        device=args.device,
        base_resnet=True,
        model_type="dinov2"
    )
    
    
    # Training setup
    num_epochs = args.epochs  # Example epoch count
    # Triplet Loss
    triplet_loss = nn.TripletMarginLoss(margin=1.0)
    # Optimizer
    optimizer = optim.Adam(embedding_model.model.parameters(), lr=0.003)
    # Scheduler
    T_max = num_epochs  # Here, we set it to the total number of epochs for one cycle
    eta_min = 0.0001  # The minimum learning rate, adjust as needed
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Training loop
    print("Starting training...")
    best_loss = float('inf')

    scaler = GradScaler()
    
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(enumerate(train_dataloader_triplet), miniters=1, total=len(train_dataloader_triplet),
                  desc=f"Epoch {epoch + 1}/{num_epochs}") as dataloader_tqdm:
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
                    anchor_embedding = embedding_model.model(anchor['input']['image'])
                    positive_embedding = embedding_model.model(positive['input']['image'])
                    negative_embedding = embedding_model.model(negative['input']['image'])

                    # Compute triplet loss
                    loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

                # Backward pass and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update tqdm bar with current loss
                dataloader_tqdm.set_postfix(loss=loss.item())

                # Log loss for each batch
                writer.add_scalar('Loss/Batch', loss.item(), epoch * len(train_dataloader_triplet) + i)
                epoch_loss += loss.item()

                if i >= 10 and args.debug:
                    break

        # Average loss for the epoch
        epoch_loss /= len(train_dataloader_triplet)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)

        # Save the last model
        last_model_path = os.path.join(args.experiment_path, f'model_last_{epoch_loss:.3f}.pth')
        torch.save(embedding_model.model.state_dict(), last_model_path)

        
        # Check if this is the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(args.experiment_path, f'model_best_{best_loss:.3f}.pth')
            torch.save(embedding_model.model.state_dict(), best_model_path)
        '''
        # Evaluation on validation set
        # Need to calculate embeddings for train and validation set
        train_embeddings = embedding_model.compute_embeddings(
            train_dataloader_single
        )
        val_embeddings = embedding_model.compute_embeddings(
            val_dataloader
        )
        train_embeddings = train_embeddings.to("cpu")
        val_embeddings = val_embeddings.to("cpu")
        print(f"TRAIN | Number of images: {train_embeddings.shape[0]} | Embedding shape: {train_embeddings.shape[1]}")
        print(f"VAL   | Number of images: {val_embeddings.shape[0]}   | Embedding shape: {val_embeddings.shape[1]}")

        # Initialize KNN
        clustering_model = ClusteringModel(
            embeddings_knowledge=train_embeddings,
            lynx_ids_knowledge=train_dataset_single.dataframe['lynx_id'].to_list(),
            n_neighbors=5,
            algorithm="brute",
            metric="minkowski"
        )
        # KNN on validation set
        clustering_model.clustering(val_embeddings)

        val_eval_metrics = EvalMetrics(
            candidates_nearest_neighbors=clustering_model.candidates_nearest_neighbors,
            lynx_id_true=val_lynx_id,
            top_k=top_k
        )

        accuracy_no_threshold = val_eval_metrics.compute_accuracy(lynx_id_predicted=clustering_model.one_knn())
        writer.add_scalar("val_accuracy_1_knn_no_threshold", accuracy_no_threshold, epoch)
        print(f"VAL | Accuracy 1-KNN: {accuracy_no_threshold}")
        '''
    print(f"Best model saved at: {best_model_path}")
    print(f"Last model saved at: {last_model_path}")
    print("Training completed. Now, start of evaluation on the model of the last epoch.")

    return
    
    # From the results on the validation set at the last epoch, we compute the ideal threshold on these data.
    # This will be used to detect new individuals on the test set.
    threshold = val_eval_metrics.get_best_threshold(
        clustering_model=clustering_model,
        min_threshold=0.0,
        max_threshold=10.0,
        step=0.1
    )

    # Final evaluation on test set
    # Since we're evaluating the model from the last epoch, we can reuse the `train_embeddings`
    test_embeddings = embedding_model.compute_embeddings(
        test_dataloader
    )
    test_embeddings = test_embeddings.to("cpu")
    print(f"TRAIN | Number of images: {train_embeddings.shape[0]} | Embedding shape: {train_embeddings.shape[1]}")
    print(f"TEST  | Number of images: {test_embeddings.shape[0]}  | Embedding shape: {test_embeddings.shape[1]}")

    # Initialize KNN
    clustering_model = ClusteringModel(
        embeddings_knowledge=train_embeddings,
        lynx_ids_knowledge=train_dataset_single.dataframe['lynx_id'].to_list(),
        n_neighbors=5,
        algorithm="brute",
        metric="minkowski"
    )
    clustering_model.clustering(test_embeddings)

    test_eval_metrics = EvalMetrics(
        candidates_nearest_neighbors=clustering_model.candidates_nearest_neighbors,
        lynx_id_true=test_lynx_id,
        top_k=top_k
    )

    accuracy_no_threshold = test_eval_metrics.compute_accuracy(lynx_id_predicted=clustering_model.one_knn())
    print(f"TEST | Accuracy 1-KNN: {accuracy_no_threshold}")

    candidates_predicted_new_individual = clustering_model.check_new_individual(
        embeddings=test_embeddings,
        candidates_predicted=clustering_model.one_knn(),
        threshold=threshold,
    )

    precision_recall = test_eval_metrics.precision_recall_individual(
        candidates_predicted=candidates_predicted_new_individual,
        individual_name="New",
        verbose=True
    )

    accuracy_threshold = test_eval_metrics.compute_accuracy(
        lynx_id_predicted=candidates_predicted_new_individual,
    )
    print(f"TEST | Accuracy 1-KNN threshold: {accuracy_threshold}")

    # CMC@k + mAP@k
    candidates_nearest_neighbors_new = clustering_model.compute_candidates_nearest_neighbors_new(
        candidates_predicted_new_individual=candidates_predicted_new_individual
    )
    test_eval_metrics.candidates_nearest_neighbors = candidates_nearest_neighbors_new
    cmc_k_mean, map_k_mean = test_eval_metrics.compute_cmc_map_metrics()
    print(f"{cmc_k_mean=}")
    print(f"{map_k_mean=}")

    def format_cmc_map(data, prefix):
        data = {prefix + "@" + str(key): value for key, value in data.items()}
        return data

    cmc_k_mean = format_cmc_map(cmc_k_mean, "cmc")
    map_k_mean = format_cmc_map(map_k_mean, "map")

    metric_dict = {
        'test_new_precision': precision_recall['precision'],
        'test_new_recall': precision_recall['recall'],
        'test_accuracy_1_knn_no_threshold': accuracy_no_threshold,
        'test_accuracy_1_knn_threshold': accuracy_threshold
    }

    writer.add_hparams(
        hparam_dict={
            'threshold': threshold
        },
        metric_dict=metric_dict | cmc_k_mean | map_k_mean
    )

    print("Evaluation completed")

    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)

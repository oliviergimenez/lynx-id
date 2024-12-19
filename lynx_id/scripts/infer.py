# lynx_id/scripts/infer/infer.py
import argparse
import os
import random
import string

import pandas as pd
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from lynx_id.data.collate import collate_single
from lynx_id.data.transformations_and_augmentations import transforms_dinov2, transforms_megadescriptor, augments_dinov2
from lynx_id.model.clustering import ClusteringModel, location_lynx_image
from lynx_id.model.embeddings import EmbeddingModel
from ..data.dataset import LynxDataset


def generate_random_lynx_id(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def create_parser():
    """Create and return the argument parser for the inference script."""
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument('--model-architecture',
                        type=str,
                        required=True,
                        choices=['resnet', 'dinov2', 'megadescriptor'],
                        default='megadescriptor',
                        help='Model architecture of the foundation model.')
    parser.add_argument('--model-weights-path', type=str, required=True, help='Path to trained model weights.')
    parser.add_argument('--image-size', type=int, default=700, help="Image size")
    parser.add_argument('--input-data',
                        type=str,
                        required=True,
                        help='Path to folder containing images (at least one image) or .csv file (columns: filepath, '
                             'optional columns: date, location).')
    parser.add_argument('--output-informations-path', type=str, required=True, help='Path to output .csv file.')
    parser.add_argument('--output-embeddings-path',
                        type=str,
                        required=True,
                        help='Path to which the embeddings of our new images will be saved')
    parser.add_argument('--knowledge-embeddings-path',
                        type=str,
                        required=True,
                        help='Path to the file containing the knowledge base of all our known individuals in '
                             '`safetensors` format (embeddings torch.tensor).')
    parser.add_argument('--knowledge-informations-path',
                        type=str,
                        required=True,
                        help='Path to a .csv file containing information about the lynx in our knowledge base.')
    parser.add_argument('--threshold',
                        type=float,
                        default=1.40,
                        help='Distance threshold at which the lynx in the image is considered to be a new individual.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='Number of images to load per batch. Choose a number 2 to the nth power. A large number '
                             'could saturate your memory and cause the inference to crash.')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='how many subprocesses to use for data loading. 0 means that the data will be loaded in '
                             'the main process. (default: 0)')
    return parser


def main(args=None):
    # Example usage of the parsed arguments
    print(f"This is the infer script.")

    # use GPU if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{DEVICE=}")

    # check presence image or csv
    files = os.listdir(args.input_data)
    extensions = ['.jpg', '.jpeg', '.png', '.csv']
    image_csv_found = False
    for file in files:
        if os.path.isfile(os.path.join(args.input_data, file)) and any(file.lower().endswith(ext) for ext in extensions):
            image_csv_found = True
            break
    if not image_csv_found:
        raise RuntimeError(f"No image files found in the directory '{args.input_data}'.")

    # dataset initialization
    transform = transforms_dinov2(image_size=args.image_size) if args.model_architecture in ["dinov2", "resnet"] \
        else transforms_megadescriptor(image_size=args.image_size)
    dataset = LynxDataset(
        folder_path_images=args.input_data,
        loader='pil',
        transform=transform,
        augmentation=augments_dinov2(image_size=args.image_size),
        probabilities=[1, 0, 0],
        mode='single',
        device=DEVICE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_single
    )

    # Load embedding model
    embedding_model = EmbeddingModel(
        model_type=args.model_architecture,
        model_path=args.model_weights_path,
        device=DEVICE
    )

    # Compute embeddings
    embeddings = embedding_model.compute_embeddings(dataloader)

    # Clustering model initialization
    clustering_model = ClusteringModel(
        embeddings_knowledge=args.knowledge_embeddings_path,
        lynx_infos_knowledge=args.knowledge_informations_path,
        n_neighbors=5,
        algorithm="brute",
        metric="cosine",
    )
    # Clustering on computed embeddings
    clustering_model.clustering(embeddings.cpu())

    # Search for new individuals
    candidates_predicted_new_individual = clustering_model.check_new_individual(
        candidates_predicted=clustering_model.one_knn(),
        threshold=args.threshold,
    )

    # Generate random lynx_id for New individuals
    is_new = []
    predicted_lynx_ids = []
    all_lynx_ids = set(clustering_model.lynx_infos_knowledge['lynx_id'].unique())
    for candidate_predicted in candidates_predicted_new_individual:
        if candidate_predicted.lynx_id == "New":
            random_lynx_id = generate_random_lynx_id(10)
            while random_lynx_id in all_lynx_ids:
                random_lynx_id = generate_random_lynx_id(10)
            predicted_lynx_ids.append(random_lynx_id)
            is_new.append(True)
            all_lynx_ids.add(random_lynx_id)
        else:
            is_new.append(False)
            predicted_lynx_ids.append(candidate_predicted.lynx_id)

    # Update of nearest neighbours if a new individual has been detected
    updated_candidates_nearest_neighbors = clustering_model.compute_candidates_nearest_neighbors_new(
        candidates_predicted_new_individual)

    # Generate csv result file
    output_results_nearest = pd.DataFrame(updated_candidates_nearest_neighbors,
                                          columns=["neighbor_1", "neighbor_2", "neighbor_3", "neighbor_4",
                                                   "neighbor_5"])
    output_results_prediction = pd.DataFrame(
        {
            "filepath": dataset.dataframe.filepath.tolist(),
            "individual_predicted": predicted_lynx_ids,
            "is_new": is_new,
            "latest_picture_individual_predicted": clustering_model.most_recent_date_lynx_id(candidates_predicted_new_individual),
            "location_closest_individual": location_lynx_image(candidates_predicted_new_individual)
        }
    )
    pd.concat([output_results_prediction, output_results_nearest], axis=1).to_csv(args.output_informations_path, index=False)

    # Save embeddings of our new images
    save_file({"embeddings": embeddings}, args.output_embeddings_path)

    print('End of inference.')


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)

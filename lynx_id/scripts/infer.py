# lynx_id/scripts/infer/infer.py
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lynx_id.data.collate import collate_single
from lynx_id.data.transformations_and_augmentations import transforms
from lynx_id.model.clustering import ClusteringModel, location_lynx_image
from lynx_id.model.embeddings import EmbeddingModel
from ..data.dataset import LynxDataset


def create_parser():
    """Create and return the argument parser for the inference script."""
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to model weights.')
    parser.add_argument('--input-data',
                        type=str,
                        required=True,
                        help='Path to folder containing images (at least one image) or .csv file (columns: filepath, '
                             'optional columns: date, location).')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output .csv file.')
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
    dataset = LynxDataset(
        folder_path_images=args.input_data,
        loader='pil',
        transform=transforms,
        probabilities=[1, 0, 0],
        mode='single',
        device='auto'
    )

    # TODO: Optimization CPU ? GPU ?
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_single
    )

    # Load embedding model
    embedding_model = EmbeddingModel(
        model_path="/gpfswork/rech/ads/uxp55sd/downloaded_model/model_best_0.512.pth",
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
        metric="minkowski",
    )
    # Clustering on computed embeddings
    clustering_model.clustering(embeddings.cpu())

    # Search for new individuals
    candidates_predicted_new_individual = clustering_model.check_new_individual(
        candidates_predicted=clustering_model.one_knn(),
        threshold=args.threshold,
    )
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
            "individual_predicted": candidates_predicted_new_individual,
            "latest_picture_individual_predicted": clustering_model.most_recent_date_lynx_id(candidates_predicted_new_individual),
            "location": location_lynx_image(candidates_predicted_new_individual)

        }
    )
    pd.concat([output_results_prediction, output_results_nearest], axis=1).to_csv(args.output_path)


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)

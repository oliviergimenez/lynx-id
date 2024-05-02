# lynx_id/scripts/infer/infer.py
import argparse
import os

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from lynx_id.data.collate import collate_single
from lynx_id.data.transformations_and_augmentations import transforms
from lynx_id.model.clustering import ClusteringModel, location_lynx_image
from lynx_id.model.embeddings import EmbeddingModel
from ..data.dataset import LynxDataset


def create_parser():
    """Create and return the argument parser for the inference script."""
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument('--knowledge-informations-path',
                        type=str,
                        required=True,
                        help='Path to a .csv file containing information about the lynx in our knowledge base.')
    parser.add_argument('--knowledge-embeddings-path',
                        type=str,
                        required=True,
                        help='Path to the file containing the knowledge base of all our known individuals in '
                             '`safetensors` format (embeddings torch.tensor).')
    parser.add_argument('--updated-informations-path',
                        type=str,
                        required=True,
                        help='Path to a csv file containing at least one "filepath" and "individual_predicted" column. '
                             'If the `filepath` is already in the knowledge csv then we update it, otherwise we add it.'
                        )
    parser.add_argument('--updated-embeddings-path',
                        type=str,
                        required=True,
                        help='Path to the embeddings corresponding to the lines in the previous csv file.')
    parser.add_argument('--new-knowledge-informations-path',
                        type=str,
                        default=None,
                        help='A possible new path for the knowledge base csv (to avoid overwriting the existing file)')
    parser.add_argument('--new-knowledge-embeddings-path',
                        type=str,
                        default=None,
                        help='A possible new path for the embeddings base safetensors (to avoid overwriting the '
                             'existing file)')

    return parser


def handle_missing_columns(df: pd.DataFrame):
    df['location'] = df.get('location', np.nan)
    df['date'] = df.get('date', np.nan)
    return df


def merge_dataframe(knowledge_df: pd.DataFrame, updated_df: pd.DataFrame):
    # Add an index column to each dataframe
    knowledge_df.reset_index(inplace=True)
    updated_df.reset_index(inplace=True)

    # Merge
    merged = pd.merge(knowledge_df, updated_df, on='filepath', how='outer', suffixes=('_A', '_B'), indicator=True)

    # Retrieval of updated indices from both the knowledge base and the results dataframe.
    # Useful for updating embeddings at a later date.
    merged_both = merged[merged['_merge'] == 'both']
    updated_indices_from_knowledge = merged_both['index_A'].tolist()
    updated_indices_from_updated = merged_both['index_B'].tolist()

    return merged, updated_indices_from_knowledge, updated_indices_from_updated


def update_information_values(knowledge_df: pd.DataFrame, merged: pd.DataFrame):
    mask_location = ~merged['location_B'].isna()
    mask_date = ~merged['date_B'].isna()

    knowledge_df['lynx_id'] = merged['individual_predicted'].combine_first(
        knowledge_df['lynx_id'])
    knowledge_df.loc[mask_date, 'date'] = merged.loc[mask_date, 'date_B'].combine_first(
        knowledge_df.loc[mask_date, 'date'])
    knowledge_df.loc[mask_location, 'location'] = merged.loc[mask_location, 'location_B'].combine_first(
        knowledge_df.loc[mask_location, 'location'])

    return knowledge_df


def add_new_lines(knowledge_df: pd.DataFrame, merged: pd.DataFrame):
    B_to_add = merged[merged['_merge'] == 'right_only']
    return pd.concat([knowledge_df, B_to_add[
        ['filepath', 'individual_predicted', 'location_B', 'date_B']].rename(
        columns={'individual_predicted': 'lynx_id', 'location_B': 'location', 'date_B': 'date'})], ignore_index=True)


def update_embeddings(knowledge_embeddings: torch.Tensor, updated_embeddings: torch.Tensor,
                      updated_indices_from_knowledge: list, updated_indices_from_updated: list):
    knowledge_embeddings[updated_indices_from_knowledge] = updated_embeddings[updated_indices_from_updated]
    remaining_indices_from_updated = [i for i in range(updated_embeddings.shape[0]) if
                                      i not in updated_indices_from_updated]
    knowledge_embeddings = torch.cat((knowledge_embeddings, updated_embeddings[remaining_indices_from_updated]), dim=0)
    return knowledge_embeddings


def save_results(knowledge_df: pd.DataFrame, knowledge_embeddings: torch.Tensor, knowledge_informations_path: str,
                 knowledge_embeddings_path):
    knowledge_df.drop('index', axis=1).to_csv(knowledge_informations_path, index=False)
    save_file({"embeddings": knowledge_embeddings}, knowledge_embeddings_path)


def main(args=None):
    # Example usage of the parsed arguments
    print(f"This is the knowledge_base script.")

    # Management of new storage path
    args.new_knowledge_informations_path = args.new_knowledge_informations_path or args.knowledge_informations_path
    args.new_knowledge_embeddings_path = args.new_knowledge_embeddings_path or args.knowledge_embeddings_path

    # Information csv
    # Load csv data
    knowledge_informations_csv = pd.read_csv(args.knowledge_informations_path)
    updated_informations_csv = pd.read_csv(args.updated_informations_path)

    # Add the "location" and "date" columns if they are not already present
    updated_informations_csv = handle_missing_columns(df=updated_informations_csv)

    # Merge dataframes
    merged, updated_indices_from_knowledge, updated_indices_from_updated = merge_dataframe(
        knowledge_df=knowledge_informations_csv,
        updated_df=updated_informations_csv
    )

    # Update knowledge_df values from updated_informations_csv (information saved in merged)
    knowledge_informations_csv = update_information_values(
        knowledge_df=knowledge_informations_csv,
        merged=merged
    )

    # Add lines from updated_informations_csv that are not present in knowledge_informations_csv
    knowledge_informations_csv = add_new_lines(
        knowledge_df=knowledge_informations_csv,
        merged=merged
    )

    # Embeddings
    # Load data
    with safe_open(args.knowledge_embeddings_path, framework="pt", device="cpu") as f:
        knowledge_embeddings = f.get_tensor("embeddings")
    with safe_open(args.updated_embeddings_path, framework="pt", device="cpu") as f:
        updated_embeddings = f.get_tensor("embeddings")

    # Update of the  embeddings matrix
    knowledge_embeddings = update_embeddings(
        knowledge_embeddings=knowledge_embeddings,
        updated_embeddings=updated_embeddings,
        updated_indices_from_knowledge=updated_indices_from_knowledge,
        updated_indices_from_updated=updated_indices_from_updated
    )

    save_results(
        knowledge_df=knowledge_informations_csv,
        knowledge_embeddings=knowledge_embeddings,
        knowledge_informations_path=args.new_knowledge_informations_path,
        knowledge_embeddings_path=args.new_knowledge_embeddings_path
    )

    print("End of knowledge_base script.")


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)

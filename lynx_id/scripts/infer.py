# lynx_id/scripts/infer/infer.py
import argparse
import os
import random
import string
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from megadetector.detection.run_detector_batch import load_and_run_detector_batch
from safetensors.torch import save_file
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from tqdm import tqdm

from lynx_id.data.collate import collate_single
from lynx_id.data.transformations_and_augmentations import transforms_dinov2, transforms_megadescriptor
from lynx_id.model.clustering import ClusteringModel, location_lynx_image
from lynx_id.model.embeddings import EmbeddingModel
from lynx_id.utils.preprocess.utils import flatten_bbox, absolute_coordinates_bbox
from ..data.dataset import LynxDataset

os.environ['WANDB_DISABLED'] = 'true'  # for megadetector

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)  # hide network error message due to wandb for megadetector

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

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
                        help='How many subprocesses to use for data loading. 0 means that the data will be loaded in '
                             'the main process. (default: 0)')
    parser.add_argument('--megadetector-model-path',
                        type=str,
                        default='/lustre/fswork/projects/rech/ads/commun/megadetector/md_v5a.0.0.pt')
    parser.add_argument('--sam-model-path',
                        type=str,
                        default="/lustre/fswork/projects/rech/ads/commun/segment_anything/sam_vit_h_4b8939.pth")
    parser.add_argument('--skip-megadetector-sam',
                        action='store_true',
                        help='If enabled, avoids images passing through megadetector and SAM.')
    return parser


def main(args=None):
    # Example usage of the parsed arguments
    print(f"{color.BOLD}{'#'*20} This is the infer script. {'#'*20}{color.END}")
    start_time_global = time.time()

    # use GPU if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{color.BOLD}{DEVICE=}{color.END}")

    # check presence image or csv
    image_csv_found = False
    image_file_names = []
    # check if the input is a CSV file
    if os.path.splitext(args.input_data)[1] == '.csv':
        image_csv_found = True
        df = pd.read_csv(args.input_data)
        image_file_names = df['filepath'].tolist()
    # check if the input is a directory containing images
    if os.path.isdir(args.input_data):
        extensions = {'.jpg', '.jpeg', '.png'}
        for file in os.listdir(args.input_data):
            if os.path.isfile(os.path.join(args.input_data, file)) and any(file.lower().endswith(ext) for ext in extensions):
                image_csv_found = True
                image_file_names.append(os.path.join(args.input_data, file))

    # raise an error if no images are found
    if not image_csv_found:
         raise RuntimeError(f"No image files found in the directory '{args.input_data}'.")

    # apply megadetector + SAM
    if not args.skip_megadetector_sam:
        print(f"{color.BOLD}Preprocessing images...{color.END}")
        start_time_preprocessing = time.time()
        print(f"{color.BOLD}Megadetector preprocessing{color.END}")
        results = load_and_run_detector_batch(
            model_file=args.megadetector_model_path,
            image_file_names=image_file_names,
            quiet=True,
            include_image_size=True,
            confidence_threshold=0.5,
        )
        df_bbox = flatten_bbox({'images': results}, add_image_without_bbox=False, verbose=False)
        df_bbox = absolute_coordinates_bbox(df_bbox)

        print(f"{color.BOLD}SAM preprocessing{color.END}")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](
            checkpoint=args.sam_model_path) \
            .to(device="cuda")
        predictor = SamPredictor(sam)

        for idx in (pbar := tqdm(range(len(df_bbox)), desc="Preprocesing images")):
            row = df_bbox.iloc[idx].to_dict()
            input_box = np.array([row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"]])

            image = Image.open(row['file'])
            image = np.array(image.convert('RGB'))
            plt.imshow(image)
            predictor.set_image(image)

            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,  # we only want the segmentation with the highest score
            )
            mask = masks[0]

            image_mask = image.copy()
            image_mask[~mask, :] = 0
            image_mask = image_mask[int(row["y"]):int(row["y"]) + int(row["height"]),
                         int(row["x"]):int(row["x"]) + int(row["width"]), :]
            image_mask_pil = Image.fromarray(image_mask)

            filename = os.path.basename(row["file"])
            args.input_data = os.path.splitext(args.input_data)[0] if not os.path.isdir(args.input_data) else args.input_data
            filepath_no_bg = f'{args.input_data}/no_bg/{filename}'

            if not os.path.exists(os.path.dirname(filepath_no_bg)):
                os.makedirs(os.path.dirname(filepath_no_bg))
            image_mask_pil.save(filepath_no_bg)

        # update `input_data`  with preprocess images
        args.input_data = f'{args.input_data}/no_bg/'
        print(f"{color.BOLD}End preprocessing (total time: {round(time.time()-start_time_preprocessing, 2)}s){color.END}")

    # dataset initialization
    transform = transforms_dinov2(image_size=args.image_size) if args.model_architecture in ["dinov2", "resnet"] \
        else transforms_megadescriptor(image_size=args.image_size)
    dataset = LynxDataset(
        folder_path_images=args.input_data,
        loader='pil',
        transform=transform,
        probabilities=[0, 0, 1],
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
            "filepath": dataset.dataframe.filepath.apply(lambda x: (Path(os.getcwd()) / x).resolve()).tolist(),
            "individual_predicted": predicted_lynx_ids,
            "is_new": is_new,
            "latest_picture_individual_predicted": clustering_model.most_recent_date_lynx_id(candidates_predicted_new_individual),
            "location_closest_individual": location_lynx_image(candidates_predicted_new_individual)
        }
    )
    pd.concat([output_results_prediction, output_results_nearest], axis=1).to_csv(args.output_informations_path, index=False)

    # Save embeddings of our new images
    save_file({"embeddings": embeddings}, args.output_embeddings_path)

    print(f"{color.BOLD}{'#'*20} End of inference (total time: {round(time.time()-start_time_global, 2)}s). {'#'*20}\n"
          f"Results written here: {args.output_informations_path} & {args.output_embeddings_path}{color.END}")


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)

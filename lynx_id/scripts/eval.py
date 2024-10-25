import argparse
import datetime
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open, save_file
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader

from lynx_id.data.collate import collate_single
from lynx_id.data.dataset import LynxDataset
from lynx_id.data.transformations_and_augmentations import transforms_dinov2, augments_dinov2, transforms_megadescriptor
from lynx_id.eval.eval import EvalMetrics
from lynx_id.model.clustering import ClusteringModel
from lynx_id.model.embeddings import EmbeddingModel

image_size = 700


def create_parser():
    """Create and return the argument parser for the training triplets script."""
    parser = argparse.ArgumentParser(description='Evaluate a model on lynx dataset.')
    parser.add_argument('--model-architecture', type=str, choices=["dinov2", "megadescriptor"], default="dinov2",
                        help="Specify the foundation model used")
    parser.add_argument('--model-weights-path', type=str, help="Path to trained model weights")
    parser.add_argument('--probabilities', metavar='N', type=str, nargs='+', default=[0, 0.5, 0.5],
                        help="Image type probabilities: [classic, crop, crop and no background]")
    parser.add_argument('--countries', metavar='N', type=str, nargs='+',
                        help="Selection of images from specified countries. If 'all', select all images")
    parser.add_argument('--image-size', type=int, default=700, help="Image size")
    parser.add_argument('--train-csv', type=str, help='Path to CSV file for training dataset', default="train")
    parser.add_argument('--val-csv', type=str, help='Path to CSV file for validation dataset', default="val")
    parser.add_argument('--test-csv', type=str, help='Path to CSV file for test dataset', default="test")
    parser.add_argument('--evaluation-results-path', type=str,
                        default='/lustre/fswork/projects/rech/ads/commun/evaluation_results',
                        help="Folder in which to save evaluation results")
    parser.add_argument('--force-compute-embeddings', action='store_true',
                        help="Enable to force the calculation of embeddings without exploiting the cache if they have already been calculated.")
    parser.add_argument('--min-threshold', type=float, default=0.0,
                        help="Minimum threshold value for optimal threshold search to detect new individuals")
    parser.add_argument('--max-threshold', type=float, default=50.0,
                        help="Maximum threshold value for optimal threshold search to detect new individuals")
    parser.add_argument('--step-threshold', type=float, default=0.001,
                        help="Step value for optimal threshold search to detect new individuals")
    parser.add_argument('--no-augmentation', action='store_true',
                        help="If enabled, load datasets without augmentation.")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser


def create_dataloader(dataset, shuffle=False, batch_size=64, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_single,
        prefetch_factor=4,
        num_workers=num_workers
    )


def print_verbose(content):
    if args.verbose:
        print(content)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

seed_everything(53)

def main(args):
    print("Running evaluation with arguments:", args)

    # Initialization of results folder
    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(args.evaluation_results_path, subdir)
    os.makedirs(results_path, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{DEVICE=}")

    # dict output initialization
    results = {
        "weights_path": args.model_weights_path,
        "image_size": args.image_size,
        "probabilities": args.probabilities,
        "countries": args.countries,
        "threshold": {
            "min_threshold": args.min_threshold,
            "max_threshold": args.max_threshold,
            "step_threshold": args.step_threshold
        },
        "no_augmentation": args.no_augmentation,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv
    }

    # Load train, val, test datasets
    probabilities = [float(p) for p in args.probabilities]
    transform = transforms_dinov2(image_size=args.image_size) if args.model_architecture == "dinov2" \
        else transforms_megadescriptor(image_size=args.image_size)
    train_dataset = LynxDataset(
        dataset_csv=Path(args.train_csv),
        countries=args.countries,
        loader="pil",
        transform=transform,
        augmentation=augments_dinov2(image_size=args.image_size) if not args.no_augmentation else None,
        probabilities=probabilities,
        mode='single',
        device=DEVICE
    )

    val_dataset = LynxDataset(
        dataset_csv=args.val_csv,
        countries=args.countries,
        loader="pil",
        transform=transform,
        augmentation=augments_dinov2(image_size=args.image_size) if not args.no_augmentation else None,
        probabilities=probabilities,
        mode='single',
        device=DEVICE
    )  # useful for computing the threshold for detecting new individuals when evaluating the test set

    test_dataset = LynxDataset(
        dataset_csv=Path(args.test_csv),
        countries=args.countries,
        loader="pil",
        transform=transform,
        augmentation=augments_dinov2(image_size=args.image_size) if not args.no_augmentation else None,
        probabilities=probabilities,
        mode='single',
        device=DEVICE
    )

    # Dataloader initialization
    train_dataloader = create_dataloader(train_dataset)
    val_dataloader = create_dataloader(val_dataset)
    test_dataloader = create_dataloader(test_dataset)

    # Model initialization
    if args.model_architecture == "dinov2":
        embedding_model = EmbeddingModel(
            model_path=args.model_weights_path,
            device=DEVICE,
            model_type='dinov2',
            custom_path='/lustre/fswork/projects/rech/ads/commun'
        )
    elif args.model_architecture == "megadescriptor":
        embedding_model = EmbeddingModel(
            model_path=args.model_weights_path,
            device=DEVICE,
            model_type='megadescriptor',
        )
    else:
        raise ValueError("Model not supported")

    if args.model_weights_path:
        filename = os.path.splitext(os.path.basename(args.model_weights_path))[0]
    else:
        filename = args.model_architecture
    print_verbose(f"{filename=}")

    embedding_model.model.eval()

    BASE_PATH = Path("/lustre/fswork/projects/rech/ads/commun/embeddings_evaluation")
    if os.path.exists(BASE_PATH / f"{filename}.safetensors") and not args.force_compute_embeddings:  # load embeddings
        with safe_open(BASE_PATH / f"{filename}.safetensors",
                       framework="pt", device="cpu") as f:
            train_embeddings = f.get_tensor("train_embeddings")
            val_embeddings = f.get_tensor("val_embeddings")
            test_embeddings = f.get_tensor("test_embeddings")
    else:  # compute embeddings
        train_embeddings = embedding_model.compute_embeddings(train_dataloader).to("cpu")
        val_embeddings = embedding_model.compute_embeddings(val_dataloader).to("cpu")
        test_embeddings = embedding_model.compute_embeddings(test_dataloader).to("cpu")

        if not os.path.exists(BASE_PATH / f"{filename}.safetensors") or args.force_compute_embeddings:
            print_verbose("Save embeddings")
            data = {
                'train_embeddings': train_embeddings,
                'val_embeddings': val_embeddings,
                'test_embeddings': test_embeddings
            }
            save_file(data, BASE_PATH / f"{filename}.safetensors")

    # update lynx_id of val and test sets for images whose lynx_id does not appear in the training set
    val_lynx_id = val_dataset.compute_new_lynx_id(train_dataset)
    test_lynx_id = test_dataset.compute_new_lynx_id(train_dataset)
    # Check that we have new individuals
    assert "New" in test_lynx_id
    assert "New" in val_lynx_id

    # Initialization Nearest Neighbors
    train_lynx_infos = train_dataset.dataframe[['lynx_id', 'date', 'location', 'filepath']].copy()
    top_k = (1, 2, 3, 4, 5)
    clustering_model = ClusteringModel(
        embeddings_knowledge=train_embeddings,
        lynx_infos_knowledge=train_lynx_infos,
        n_neighbors=5,
        algorithm="brute",
        metric="minkowski",
    )

    # Validation set
    results_validation = {}
    print(f"{'*' * 50} Process validation set {'*' * 50}")
    clustering_model.clustering(val_embeddings)
    val_eval_metrics = EvalMetrics(
        candidates_nearest_neighbors=clustering_model.candidates_nearest_neighbors,
        lynx_id_true=val_lynx_id,
        top_k=top_k
    )

    # accuracy without detection of New individual
    accuracy_no_threshold = val_eval_metrics.compute_accuracy(lynx_id_predicted=clustering_model.one_knn())
    print_verbose(f"Accuracy 1-KNN: {accuracy_no_threshold}")
    results_validation["accuracy_no_threshold"] = accuracy_no_threshold

    # compute best_threshold for validation set, it will be use for the test set
    val_best_threshold = val_eval_metrics.get_best_threshold(
        clustering_model=clustering_model,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step_threshold
    )
    print_verbose(f"{val_best_threshold=}")
    results_validation["best_threshold"] = val_best_threshold
    results["validation"] = results_validation

    # Test set
    results_test = {}
    print(f"{'*' * 50} Process test set {'*' * 50}")
    clustering_model.clustering(test_embeddings)
    test_eval_metrics = EvalMetrics(
        candidates_nearest_neighbors=clustering_model.candidates_nearest_neighbors,
        lynx_id_true=test_lynx_id,
        top_k=top_k
    )

    accuracy_no_threshold = test_eval_metrics.compute_accuracy(lynx_id_predicted=clustering_model.one_knn())
    print_verbose(f"Accuracy 1-KNN no threshold : {accuracy_no_threshold}")
    results_test["accuracy_no_threshold"] = accuracy_no_threshold

    accuracy_no_threshold_no_new = test_eval_metrics.compute_accuracy(lynx_id_predicted=clustering_model.one_knn(),
                                                                      no_new=True)
    print_verbose(f"Accuracy 1-KNN no threshold + no new : {accuracy_no_threshold_no_new}")
    results_test["accuracy_no_threshold_no_new"] = accuracy_no_threshold_no_new

    theorical_best_threshold = test_eval_metrics.get_best_threshold(
        clustering_model=clustering_model,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step_threshold
    )
    print_verbose(f"{theorical_best_threshold=}")
    results_test["best_threshold"] = theorical_best_threshold

    ### Threshold ###
    print_verbose(f"{'-' * 50} New individual based on threshold {'-' * 50}")
    print_verbose(f"Threshold used: {val_best_threshold}\n")
    results_test["threshold_used"] = val_best_threshold
    candidates_predicted_new_individual = clustering_model.check_new_individual(
        embeddings=test_embeddings,
        candidates_predicted=clustering_model.one_knn(),
        threshold=val_best_threshold,
    )

    precision_recall = test_eval_metrics.precision_recall_individual(
        candidates_predicted=candidates_predicted_new_individual,
        individual_name="New",
        verbose=True
    )
    print_verbose(f"{precision_recall=}")
    results_test["precision_new"] = precision_recall["precision"]
    results_test["recall_new"] = precision_recall["recall"]

    # CMC@k + mAP@k
    candidates_nearest_neighbors_new = clustering_model.compute_candidates_nearest_neighbors_new(
        candidates_predicted_new_individual=candidates_predicted_new_individual
    )
    test_eval_metrics.candidates_nearest_neighbors = candidates_nearest_neighbors_new
    cmc_k_mean, map_k_mean = test_eval_metrics.compute_cmc_map_metrics()
    print_verbose(f"{cmc_k_mean=}")
    print_verbose(f"{map_k_mean=}")
    results_test["cmc@k"] = cmc_k_mean
    results_test["map@k"] = map_k_mean

    # unbalanced evaluation
    y_true = test_eval_metrics.lynx_id_true
    y_pred = [candidate.lynx_id for candidate in candidates_predicted_new_individual]
    cm = confusion_matrix(y_true, y_pred)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)
        per_class = [(value, row_sum) for value, row_sum in zip(per_class, cm.sum(axis=1))]

    per_class_df = pd.DataFrame(per_class, columns=['accuracy', 'number of images'])
    print_verbose(per_class_df)

    per_class_score = [score[0] for score in per_class]

    balanced_accuracy = np.mean(np.nan_to_num(per_class_score, nan=0))
    print_verbose(f"{balanced_accuracy=}")
    results_test["balanced_accuracy"] = balanced_accuracy

    precision_unweighted = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_unweighted = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print_verbose(f"Precision score (unweighted/macro): {precision_unweighted}")
    print_verbose(f"Recall score (unweighted/macro): {recall_unweighted}")
    results_test["precision_unweighted"] = precision_unweighted
    results_test["recall_unweighted"] = recall_unweighted

    results["test"] = results_test

    with open(f"/{results_path}/{filename}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("End evaluation")


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)

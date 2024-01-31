import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from oml.functional.metrics import calc_cmc, calc_map, calc_precision
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from data_pipeline.dataset import LynxDataset
from data_pipeline.transformations_and_augmentations import transforms
from data_pipeline.triplets import collate_single

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}")

# Load our model
model_weights = torch.load("/gpfswork/rech/ads/commun/kg_tests/model_best_0.512.pth", map_location=DEVICE)
model = models.resnet50(pretrained=False)
model.fc = nn.Identity()
model.load_state_dict(model_weights)
model.to(DEVICE)

# Load the dataset
lynxDataset = LynxDataset(
    Path("/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_france.csv"),
    transform=transforms,
)
dataloader = DataLoader(lynxDataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_single)

# Compute embeddings
embeddings = None
for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
    batch_tensor = torch.stack(batch[0]['image']).to(DEVICE).float()

    with torch.no_grad():
        batch_embeddings = model(batch_tensor)

        if embeddings is None:
            embeddings = batch_embeddings
        else:
            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

# # Selection of N random images to be identified later
# candidates_indices = torch.randperm(len(lynxDataset))[:100]

# Select n images (index) from the dataset. The individual associated with the images must have at least 20 images
# (threshold set experimentally).
n = 100
min_img_per_individual = 20
candidates_indices = set()
data = pd.read_csv("/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_france.csv")
lynx_id_counts = data['lynx_id'].value_counts()

while len(candidates_indices) < n:
    indice = random.randint(0, len(lynxDataset)-1)
    if lynx_id_counts[lynxDataset[indice][1]['lynx_id']] >= min_img_per_individual:
        candidates_indices.add(indice)
candidates_indices = torch.tensor(list(candidates_indices))
print(f"{candidates_indices=}")


# Get their associated embeddings
embeddings_candidates = embeddings[candidates_indices].to("cpu")
print(embeddings_candidates.shape)

mask_knowledge = torch.ones(embeddings.size(0), dtype=torch.bool)
mask_knowledge[candidates_indices] = False
embeddings_knowledge = embeddings[mask_knowledge].to("cpu")
print(embeddings_knowledge.shape)

# Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric="minkowski").fit(embeddings_knowledge)
distances, indices = neighbors.kneighbors(embeddings_candidates)

candidates_nearest_neighbors = []
for nearest_indices in indices:
    tmp = []
    for indice in nearest_indices:
        tmp.append(lynxDataset[indice][1]['lynx_id'])
    candidates_nearest_neighbors.append(tmp)
print(f"{candidates_nearest_neighbors=}")

# True lynx_id
candidates_id = []
for indice in candidates_indices:
    candidates_id.append(lynxDataset[indice.item()][1]['lynx_id'])
print(f"{candidates_id=}")

# n-knn
candidates_predicted_n_knn = [Counter(candidate).most_common(1)[0][0] for candidate in candidates_nearest_neighbors]
print(f"{candidates_predicted_n_knn=}")

# 1-knn
candidates_predicted_1_knn = [candidate[0] for candidate in candidates_nearest_neighbors]
print(f"{candidates_predicted_1_knn=}")


# Compute some metrics
def compute_accuracy(candidates_predicted, candidates_refs):
    correct_predictions = sum(p == r for p, r in zip(candidates_predicted, candidates_refs))

    total_predictions = len(candidates_predicted)
    accuracy = correct_predictions / total_predictions

    return accuracy


accuracy_1_knn = compute_accuracy(candidates_predicted_1_knn, candidates_id)
print(f"{accuracy_1_knn=}")
accuracy_n_knn = compute_accuracy(candidates_predicted_n_knn, candidates_id)
print(f"{accuracy_n_knn=}")

# TODO: consider new individuals

# CMC@k, mAP@k, precision@k
top_k = (1, 2, 3, 4, 5)

candidates_acc_k_list = [[1 if candidate == candidate_id else 0 for candidate in candidates_row] for
                         candidates_row, candidate_id in zip(candidates_nearest_neighbors, candidates_id)]
candidates_acc_k_tensor = torch.tensor(candidates_acc_k_list, dtype=torch.bool)


def compute_mean_per_top_k(metric_output):
    metric_mean = torch.mean(torch.stack(metric_output), dim=1)
    return {k: v.item() for k, v in zip(top_k, metric_mean)}


# CMC@k
cmc_k = calc_cmc(candidates_acc_k_tensor, top_k)
cmc_k_mean = compute_mean_per_top_k(cmc_k)
print(f"{cmc_k_mean=}")

# mAP@k
n_gt = torch.tensor([100])
map_k = calc_map(candidates_acc_k_tensor, n_gt=n_gt, top_k=top_k)
map_k_mean = compute_mean_per_top_k(map_k)
print(f"{map_k_mean=}")

# Precision@k
precision_k = calc_precision(candidates_acc_k_tensor, n_gt=n_gt, top_k=top_k)
precision_k_mean = compute_mean_per_top_k(precision_k)
print(f"{precision_k_mean=}")


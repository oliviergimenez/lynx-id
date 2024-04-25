from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class ClusteringModel:
    def __init__(self, embeddings_knowledge: torch.Tensor | str, lynx_ids_knowledge: List[str] | str,
                 n_neighbors: int = 5, algorithm: str = 'brute', metric: str = 'minkowski'):
        self.embeddings_knowledge = embeddings_knowledge
        if isinstance(embeddings_knowledge, str):
            self.load_safetensors()

        self.lynx_ids_knowledge = lynx_ids_knowledge
        if isinstance(self.lynx_ids_knowledge, str):
            self.load_lynx_ids()

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric

        self.nearest_neighbors = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            metric=self.metric,
        ).fit(self.embeddings_knowledge)

        self.distances = None
        self.indices = None
        self.cluster_variances = None
        self.cluster_means = None
        self.candidates_nearest_neighbors = None

    def load_safetensors(self):
        with safe_open(self.embeddings_knowledge, framework="pt", device="cpu") as f:
            self.embeddings_knowledge = f.get_tensor("embeddings")

    def load_lynx_ids(self):
        self.lynx_ids_knowledge = pd.read_csv(self.lynx_ids_knowledge)['lynx_id'].tolist()

    def clustering(self, embeddings_candidates: torch.tensor):
        self.distances, self.indices = self.nearest_neighbors.kneighbors(embeddings_candidates)

        self.candidates_nearest_neighbors = [
            [self.lynx_ids_knowledge[indice] for indice in nearest_indices]
            for nearest_indices in self.indices
        ]

        return self.candidates_nearest_neighbors

    def one_knn(self):
        return [candidate[0] for candidate in self.candidates_nearest_neighbors]

    def n_knn(self):
        return [Counter(candidate).most_common(1)[0][0] for candidate in self.candidates_nearest_neighbors]

    def compute_candidates_nearest_neighbors_new(self, candidates_predicted_new_individual,
                                                 update_candidates_nearest_neighbors: bool = False):
        updated = [
            ["New"] + neighbors[0:-1] if candidate == "New" else neighbors
            for candidate, neighbors in zip(candidates_predicted_new_individual, self.candidates_nearest_neighbors)
        ]
        if update_candidates_nearest_neighbors:
            self.candidates_nearest_neighbors = updated

        return updated

    def check_new_individual(self, candidates_predicted: List[str], embeddings: torch.Tensor = None,
                             success_percentage_threshold: int = 100, threshold: float = None,
                             confidence: float = 0.997):
        candidates_predicted_local = candidates_predicted.copy()
        distances = self.distances[:, 0]
        # Threshold
        if threshold is not None and distances is not None:
            candidates_predicted_local = [
                "New" if distance > threshold else candidate
                for candidate, distance in zip(candidates_predicted, distances)
            ]

        # Gaussian
        else:
            confidence_intervals = self.compute_confidence_intervals(
                confidence=confidence
            )
            for index, (candidate_dot, candidate_predicted) in enumerate(zip(embeddings, candidates_predicted)):

                if candidate_predicted in confidence_intervals.keys():
                    confidence_intervals_for_candidate = confidence_intervals[candidate_predicted]

                    num_successful_cases = sum(
                        interval[0] <= dot <= interval[1]
                        for dot, interval in zip(candidate_dot, confidence_intervals_for_candidate)
                    )
                    min_successful_cases = int((candidate_dot.shape[0] * success_percentage_threshold) / 100)
                    in_confidence_interval = num_successful_cases >= min_successful_cases

                else:
                    in_confidence_interval = torch.tensor(True)

                if in_confidence_interval.item() is False:
                    candidates_predicted_local[index] = "New"

        return candidates_predicted_local

    def compute_cluster_means_variances(self):
        self.cluster_variances = {}
        self.cluster_means = {}
        unique_knowledge_lynx_id = set(self.lynx_ids_knowledge)
        for lynx in tqdm(unique_knowledge_lynx_id):
            # get all embeddings for this lynx
            lynx_idx = [i for i in range(len(self.lynx_ids_knowledge)) if self.lynx_ids_knowledge[i] == lynx]
            self.cluster_variances[lynx] = torch.var(self.embeddings_knowledge[lynx_idx], dim=0).tolist()
            self.cluster_means[lynx] = torch.mean(self.embeddings_knowledge[lynx_idx], dim=0).tolist()

    def compute_confidence_intervals(self, confidence, force_recompute_cluster_stats=False):
        if (self.cluster_variances is None and self.cluster_variances is None) or force_recompute_cluster_stats is True:
            self.compute_cluster_means_variances()
        intervals = {}

        for (lynx, mean), variance in zip(tqdm(self.cluster_means.items(), desc="compute_confidence_intervals"),
                                          self.cluster_variances.values()):  # N clusters (lynx) in our knowledge base
            interval = norm.interval(confidence, loc=mean, scale=np.sqrt(variance))
            interval_formatted = tuple((element[0], element[1]) for element in zip(interval[0], interval[1]))
            intervals[lynx] = interval_formatted

        return intervals

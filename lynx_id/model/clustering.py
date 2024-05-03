from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


@dataclass
class LynxImage:
    filepath: str | None
    lynx_id: str  # always mandatory
    location: str | None
    date: datetime | None

    def __str__(self):
        return self.lynx_id

    def __repr__(self):
        return self.__str__()


def location_lynx_image(lynx_images: list[LynxImage]):
    return [lynx_img.location for lynx_img in lynx_images]


class ClusteringModel:
    def __init__(self, embeddings_knowledge: torch.Tensor | str, lynx_infos_knowledge: pd.DataFrame | str,
                 n_neighbors: int = 5, algorithm: str = 'brute', metric: str = 'minkowski'):
        self.embeddings_knowledge = embeddings_knowledge
        if isinstance(embeddings_knowledge, str):
            self.load_safetensors()

        self.lynx_infos_knowledge = lynx_infos_knowledge
        if isinstance(self.lynx_infos_knowledge, str):
            self.load_lynx_infos()
        self.lynx_infos_knowledge['date'] = pd.to_datetime(self.lynx_infos_knowledge['date'], format="ISO8601")

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
        self.candidates_predicted = None

    def load_safetensors(self):
        with safe_open(self.embeddings_knowledge, framework="pt", device="cpu") as f:
            self.embeddings_knowledge = f.get_tensor("embeddings")

    def load_lynx_infos(self):
        self.lynx_infos_knowledge = pd.read_csv(self.lynx_infos_knowledge)

    def clustering(self, embeddings_candidates: torch.tensor):
        self.distances, self.indices = self.nearest_neighbors.kneighbors(embeddings_candidates)

        self.candidates_nearest_neighbors = [
            [LynxImage(filepath=self.lynx_infos_knowledge.iloc[indice]['filepath'],
                       lynx_id=self.lynx_infos_knowledge.iloc[indice]['lynx_id'],
                       location=self.lynx_infos_knowledge.iloc[indice]['location'],
                       date=self.lynx_infos_knowledge.iloc[indice]['date'])
             for indice in nearest_indices]
            for nearest_indices in self.indices
        ]

        return self.candidates_nearest_neighbors

    def one_knn(self):
        prediction_lynx_img = [candidates[0] for candidates in self.candidates_nearest_neighbors]

        return prediction_lynx_img

    def n_knn(self):
        prediction_lynx_img = [max(candidates, key=lambda x: [lynx_img.lynx_id
                                                              for lynx_img in candidates].count(x.lynx_id))
                               for candidates in self.candidates_nearest_neighbors]

        return prediction_lynx_img

    def compute_candidates_nearest_neighbors_new(self, candidates_predicted_new_individual,
                                                 update_candidates_nearest_neighbors: bool = False):
        updated = [
            [candidate] + neighbors[0:-1] if candidate.lynx_id == "New" else neighbors
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
                LynxImage(filepath=None, lynx_id="New", location=None, date=None) if distance > threshold else candidate
                for candidate, distance in zip(candidates_predicted, distances)
            ]

        # Gaussian
        else:
            confidence_intervals = self.compute_confidence_intervals(
                confidence=confidence
            )
            for index, (candidate_dot, candidate_predicted) in enumerate(zip(embeddings, candidates_predicted)):

                if candidate_predicted.lynx_id in confidence_intervals.keys():
                    confidence_intervals_for_candidate = confidence_intervals[candidate_predicted.lynx_id]

                    num_successful_cases = sum(
                        interval[0] <= dot <= interval[1]
                        for dot, interval in zip(candidate_dot, confidence_intervals_for_candidate)
                    )
                    min_successful_cases = int((candidate_dot.shape[0] * success_percentage_threshold) / 100)
                    in_confidence_interval = num_successful_cases >= min_successful_cases

                else:
                    in_confidence_interval = torch.tensor(True)

                if in_confidence_interval.item() is False:
                    candidates_predicted_local[index] = LynxImage(filepath=None,
                                                                  lynx_id="New",
                                                                  location=None,
                                                                  date=None)

        self.candidates_predicted = candidates_predicted_local
        return candidates_predicted_local

    def compute_cluster_means_variances(self):
        self.cluster_variances = {}
        self.cluster_means = {}
        lynx_ids_knowledge = self.lynx_infos_knowledge['lynx_id'].tolist()
        unique_knowledge_lynx_id = set(lynx_ids_knowledge)
        for lynx in tqdm(unique_knowledge_lynx_id):
            # get all embeddings for this lynx
            lynx_idx = [i for i in range(len(lynx_ids_knowledge)) if lynx_ids_knowledge[i] == lynx]
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

    def most_recent_date_lynx_id(self, lynx_images: list[LynxImage]):
        latest_date_per_lynx_id = self.lynx_infos_knowledge.groupby('lynx_id')['date'].max()
        latest_dates = [latest_date_per_lynx_id.get(lynx_img.lynx_id, None) for lynx_img in lynx_images]
        return latest_dates

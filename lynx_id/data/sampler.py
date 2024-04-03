from collections import defaultdict

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.class_counts = defaultdict(int)

        # Count instances per class
        for _, output in dataset:
            lynx_id = output['lynx_id']
            self.class_counts[lynx_id] += 1

        # Calculate weights for each instance
        self.weights = [1.0 / self.class_counts[dataset[idx][1]['lynx_id']] for idx in self.indices]

    def __iter__(self):
        # Sample anchors based on weights
        anchors = torch.multinomial(torch.tensor(self.weights, dtype=torch.double), len(self.weights), replacement=True)
        return (self.indices[i] for i in anchors)

    def __len__(self):
        return len(self.indices)

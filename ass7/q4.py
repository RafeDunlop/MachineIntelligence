import random
import numpy as np

class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size, seed=0):
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        n = len(self.dataset)
        indices = random.choices(range(n), weights=self.weights, k=self.sample_size)
        return self.dataset[indices]
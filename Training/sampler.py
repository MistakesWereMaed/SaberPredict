import torch

from torch.utils.data import WeightedRandomSampler

class ClassWeightedSampler:
    def __init__(self, weights, labels):
        sample_weights = torch.tensor(
            [weights[label] for label in labels],
            dtype=torch.double
        )

        # Create WeightedRandomSampler
        self.sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def get_sampler(self):
        return self.sampler
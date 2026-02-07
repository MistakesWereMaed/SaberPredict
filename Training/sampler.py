import torch
import numpy as np

MINORITY_WEIGHT = 4
BOUNDARY_WEIGHT = 2

class BoundaryAwareSequenceSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        sequence_df,
        generator=None
    ):
        self.sequence_df = sequence_df.reset_index(drop=True)

        weights = np.ones(len(sequence_df))
        weights[sequence_df["has_minority"]] *= MINORITY_WEIGHT
        weights[sequence_df["has_boundary"]] *= BOUNDARY_WEIGHT

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.generator = generator

    def __len__(self):
        return len(self.sequence_df)

    def __iter__(self):
        indices = torch.multinomial(
            self.weights,
            num_samples=len(self.weights),
            replacement=True,
            generator=self.generator
        )
        return iter(indices.tolist())
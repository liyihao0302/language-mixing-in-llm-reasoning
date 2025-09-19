import torch
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random

class ClassBalancedBatchSampler(Sampler):
    """
    Custom sampler that ensures each minibatch contains at least one sample from each class.
    Assumes 3-class classification (0: harmful, 1: neutral, 2: helpful).
    """

    def __init__(self, labels, batch_size, drop_last=False):
        """
        Args:
            labels (List[int] or np.ndarray): List of class labels.
            batch_size (int): Batch size.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.label_to_indices = defaultdict(list)

        # Group indices by label
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.num_classes = len(self.label_to_indices)
        assert self.batch_size >= self.num_classes, "Batch size must be >= number of classes."

        # Convert lists to numpy arrays for fast sampling
        for label in self.label_to_indices:
            self.label_to_indices[label] = np.array(self.label_to_indices[label])

    def __iter__(self):
        all_batches = []
        total_samples = len(self.labels)
        num_batches = total_samples // self.batch_size

        for _ in range(num_batches):
            batch = []

            # Sample one from each class first
            for label in range(self.num_classes):
                label_indices = self.label_to_indices[label]
                idx = np.random.choice(label_indices)
                batch.append(idx)

            # Fill the rest of the batch randomly
            remaining_size = self.batch_size - self.num_classes
            all_indices = np.arange(total_samples)
            extras = np.random.choice(all_indices, size=remaining_size, replace=False)
            batch.extend(extras)

            np.random.shuffle(batch)
            all_batches.append(batch)

        if not self.drop_last:
            leftover = total_samples % self.batch_size
            if leftover > 0:
                all_batches.append(list(np.random.choice(np.arange(total_samples), size=leftover, replace=False)))

        return iter(all_batches)

    def __len__(self):
        return len(self.labels) // self.batch_size

# Example usage:
# sampler = ClassBalancedBatchSampler(y_train, batch_size=64)
# loader = DataLoader(train_dataset, batch_sampler=sampler)


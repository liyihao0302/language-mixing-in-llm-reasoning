# src/probe/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Implements Supervised Contrastive Loss from:
    "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [batch_size, embed_dim] (must be normalized)
            labels: Tensor of shape [batch_size] with int class labels

        Returns:
            A scalar contrastive loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features (should already be normalized, but just in case)
        features = F.normalize(features, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask to exclude self-comparisons
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(self_mask, -1e9)

        # Get label matches (positive pairs)
        labels = labels.contiguous().view(-1, 1)
        positive_mask = (labels == labels.T) & ~self_mask

        # Compute log prob
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Masked sum of log-probs over positives
        positive_log_prob = (log_prob * positive_mask).sum(dim=1)
        positive_count = positive_mask.sum(dim=1).clamp(min=1)

        loss = -positive_log_prob / positive_count
        return loss.mean()

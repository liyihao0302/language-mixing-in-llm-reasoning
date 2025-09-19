# src/probe/train_supcon.py

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Tuple
import torch.nn.functional as F

from probe.sampler import ClassBalancedBatchSampler
from probe.model import get_model
from probe.evaluate import evaluate_predictions, compute_decision_utility
from probe.contrastive_loss import SupervisedContrastiveLoss
from probe.prepare_data import (
    load_all_raw_activations,
    stratified_group_split_by_dataset,
    apply_pca_split
)

class ContrastiveDataset(Dataset):
    """PyTorch dataset for supervised contrastive probing."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, dist):
        self.features = torch.tensor(features).float()
        self.labels = torch.tensor(labels).long()
        self.dist = torch.tensor(dist).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.dist[idx]  # Return soft label


utility_case1 = torch.tensor([
    [ 1.0, 0.0,  0.0],
    [ 0.0, 0.0,  0.0],
    [-1.0, 0.0,  0.0],
])  # natural switch

utility_case2 = torch.tensor([
    [ 0.0,  0.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 0.0,  0.0, +1.0],
])  # no natural switch

def decision_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    is_natural: torch.BoolTensor,
) -> torch.Tensor:
    """
    logits:     [B,3]  raw scores for (H,N,B)
    labels:     [B]    ground-truth in {0,1,2}
    is_natural: [B]    True if this example uses case1, False→case2
    returns:    scalar = - mean expected ΔU
    """
    # 1) probabilities
    probs = F.softmax(logits, dim=1)            # [B,3]
    
    # 2) gather the correct ΔU‐row for each example under both cases
    #    each is [B,3]
    du1 = utility_case1[labels]                  
    du2 = utility_case2[labels]
    
    # 3) select per‐example ΔU rows based on is_natural
    #    you can do it with a mask:
    du = torch.where(
        is_natural.unsqueeze(1),
        du1,
        du2
    )  # [B,3]
    
    # 4) expected utility per sample
    eu = (probs * du).sum(dim=1)                # [B]
    
    # 5) loss = - average expected utility
    return -eu.mean()

def train_supcon_probe(
    dataset_configs: List[Tuple[str, str, str]],
    layers: List[int],
    pca_dim: int,
    save_path: str,
    input_dim: int = 256,
    hidden_dim: int = 256,
    projection_dim: int = 128,
    use_metadata: bool = True,
    alpha: float = 0.5,
    temperature: float = 0.1,
    class_weights: List[float] = [5.0, 0.2, 5.0],
    batch_size: int = 64,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[
    nn.Module,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[PCA, StandardScaler] 
]:
    """
    Train a supervised contrastive probe.

    Returns:
        - Trained PyTorch model
        - Validation logits
        - Validation labels
        - Validation natural switch flags
        - (X_test, y_test, df_test): raw test split for downstream use
    """

    # === Step 1: Load and preprocess dataset ===
    print("Loading raw activations and metadata...")
    X_raw, y_raw, df, dist = load_all_raw_activations(
        dataset_configs=dataset_configs,
        layers=layers,
        use_metadata=use_metadata
    )

    print("Splitting into train/val/test by (dataset_src, problem_id)...")
    train_mask, val_mask, test_mask = stratified_group_split_by_dataset(
        df, val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state
    )

    print("Applying PCA fit on training activations only...")
    X_train, X_val, X_test, pca, scaler = apply_pca_split(
        X_raw, df, train_mask, val_mask, test_mask, pca_dim=pca_dim
    )
    
    #X_train, X_val, X_test = X_raw[train_mask], X_raw[val_mask], X_raw[test_mask]
    y_train, y_val, y_test = y_raw[train_mask], y_raw[val_mask], y_raw[test_mask]
    dist_train, dist_val, dist_test = dist[train_mask], dist[val_mask], dist[test_mask]
    #pca, scaler = None, None  # Placeholder for PCA and scaler if not used
    print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    val_df = df[val_mask].reset_index(drop=True)
    val_is_natural = val_df["if_natural"].astype(bool).values


    # === Step 2: DataLoader setup ===
    train_ds = ContrastiveDataset(X_train, y_train, dist_train)
    val_ds = ContrastiveDataset(X_val, y_val, dist_val)
    sampler = ClassBalancedBatchSampler(y_train, batch_size=batch_size)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # === Step 3: Initialize model and losses ===
    model = get_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=3,
        projection_dim=projection_dim,
        model_type="supcon_dual"
    ).to(device)

    contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # === Step 4: Training loop ===
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        total_count = 0
        for xb, yb, distb in train_loader:
            #import pdb; pdb.set_trace()  # Debugging breakpoint
            xb, yb, distb = xb.to(device), yb.to(device), distb.to(device)
            z, logits = model(xb)
            #loss = alpha * contrastive_loss(z, yb) + (1 - alpha) * ce_loss(logits, yb)
            loss = decision_loss(logits, yb, xb[..., -1].bool()) + 0.5 * ce_loss(logits, distb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        print(f"[Epoch {epoch:02d}] Train Loss: {total_loss/ total_count:.4f}")

        # === Step 5: Validation ===
        model.eval()
        all_preds, all_labels, val_logits = [], [], []
        with torch.no_grad():
            for xb, yb, distb in val_loader:
                _, logits = model(xb.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
                val_logits.append(logits.cpu().numpy())

        print(f"[Epoch {epoch:02d}] Validation Accuracy:")
        evaluate_predictions(np.array(all_labels), np.array(all_preds))
        print('-----Natural Switch Accuracy:------')
        evaluate_predictions(np.array(all_labels)[val_is_natural==1], np.array(all_preds)[val_is_natural==1])
        
        print('-----Synthetic Switch Accuracy:------')
        evaluate_predictions(np.array(all_labels)[val_is_natural==0], np.array(all_preds)[val_is_natural==0])
        
        utility = compute_decision_utility(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            is_natural=val_is_natural
        )
        print(f"[Epoch {epoch:02d}] Decision Utility: {utility:.4f}")

    # === Step 6: Save model checkpoint ===
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")

    return (
        model,
        np.vstack(val_logits),
        np.array(all_labels),
        val_is_natural,
        (X_test, y_test, df[test_mask].reset_index(drop=True)),
        (pca, scaler)  
    )
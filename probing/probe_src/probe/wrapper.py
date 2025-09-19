import torch
import numpy as np
from torch.nn.functional import softmax
import torch.nn.functional as F
from probe.model import get_model

class ProbeWrapper:
    """
    Wrapper for inference-time use of a trained SupCon probe.
    Applies PCA + meta features + threshold tuning + probe prediction.
    """
    def __init__(
        self,
        checkpoint_path: str,
        pca,
        scaler,
        layers: list,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        threshold=(0.5, 0.5),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.pca = pca
        self.scaler = scaler
        self.layers = layers

        # Initialize model and load weights
        self.model = get_model(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            model_type="supcon"
        ).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

        # (τ_h, τ_help)
        self.threshold = threshold

    def __call__(self, activations: dict, meta_features: dict) -> torch.Tensor:
        """
        Run probe prediction on test-time activations and meta features.

        Args:
            activations (dict): {layer_idx (str) -> [B, D]} raw activation tensors
            meta_features (dict): {name -> [B] float tensor} metadata features

        Returns:
            torch.Tensor: predicted class labels (0: harmful, 1: neutral, 2: beneficial)
        """
        # === Step 1: Concatenate raw activations for the configured layers ===
        act_list = []
        for layer in self.layers:
            layer_key = str(layer)
            if layer_key not in activations:
                raise ValueError(f"Missing activations for layer {layer_key}")
            act_list.append(activations[layer_key].clone().detach().cpu().numpy())
        X_act = np.concatenate(act_list, axis=-1)  # [B, D]

        # === Step 2: Apply PCA and standardization ===
        X_scaled = self.scaler.transform(X_act)
        X_pca = self.pca.transform(X_scaled)  # [B, pca_dim]

        # === Step 3: Concatenate metadata features ===
        meta = np.stack([
            meta_features["heuristic"].cpu().numpy(),
            meta_features["if_en_to_ch"].cpu().numpy(),
            meta_features["if_natural"].cpu().numpy(),
        ], axis=1)  # [B, 3]

        X_full = np.concatenate([X_pca, meta], axis=1)
        X_tensor = torch.tensor(X_full, dtype=torch.float32).to(self.device)

        # === Step 4: Run probe forward pass ===
        with torch.no_grad():
            _, logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        # === Step 5: Threshold-based decision ===
        if self.threshold is None:
            # Use default thresholds
            preds = torch.argmax(probs, dim=1)
        else:
            tau_h, tau_help = self.threshold
            preds = []
            for p in probs:
                if p[0] > tau_h:
                    preds.append(0)  # harmful
                elif p[2] > tau_help:
                    preds.append(2)  # beneficial
                else:
                    preds.append(1)  # neutral

        return torch.tensor(preds, device=self.device)

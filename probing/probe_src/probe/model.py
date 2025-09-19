# src/probe/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleProbe(nn.Module):
    """
    A simple multilayer perceptron for probing activations.
    Predicts one of three classes: harmful, neutral, or helpful.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3):
        super(SimpleProbe, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SupConProbe(nn.Module):
    """
    A probe with shared encoder for supervised contrastive learning and classification.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3, projection_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, projection_dim),
            #nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Linear(projection_dim, projection_dim),
        )
        #import pdb; pdb.set_trace()
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(projection_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        # Encoder output
        z = self.encoder(x)  # [B, projection_dim]
        z_normalized = F.normalize(z, dim=1)

        # Class logits from classifier head
        logits = self.classifier(z)
        return z_normalized, logits
    
class SupConDualProbe(nn.Module):
    """
    A probe with two separate encoder+classifier paths.
      - If x[..., -1] == 1: goes through the “natural” path
      - Else: goes through the “other” path
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 3,
        projection_dim: int = 128
    ):
        super().__init__()
        # both paths use the same architecture, but separate weights
        def make_path():
            return nn.Sequential(
                nn.Linear(input_dim - 1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, projection_dim),
            )
        
        self.encoder_nat = make_path()
        self.encoder_oth = make_path()
        
        # separate classifiers on top of each projection
        self.classifier_nat = nn.Sequential(
            nn.ReLU(),
            nn.Linear(projection_dim, output_dim)
        )
        self.classifier_oth = nn.Sequential(
            nn.ReLU(),
            nn.Linear(projection_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, input_dim], where x[:, -1] is a 0/1 flag:
           1 → “natural” path; 0 → “other” path.
        Returns (z_normalized, logits), both [B, …].
        """
        flag = x[:, -1].bool()           # shape [B]
        x_feat = x[:, :-1]               # strip off the flag
        
        B = x.size(0)
        device = x.device
        
        # placeholders
        z = torch.zeros(B, self.encoder_nat[-1].out_features, device=device)
        logits = torch.zeros(B, self.classifier_nat[-1].out_features, device=device)
        
        # natural path
        if flag.any():
            z_nat = self.encoder_nat(x_feat[flag])
            z_norm_nat = F.normalize(z_nat, dim=1)
            logits_nat = self.classifier_nat(z_nat)
            
            z[flag] = z_nat
            logits[flag] = logits_nat
        
        # other path
        if (~flag).any():
            z_oth = self.encoder_oth(x_feat[~flag])
            z_norm_oth = F.normalize(z_oth, dim=1)
            logits_oth = self.classifier_oth(z_oth)
            
            z[~flag] = z_oth
            logits[~flag] = logits_oth
        
        # normalize full batch
        z_normalized = F.normalize(z, dim=1)
        return z_normalized, logits



def get_model(
    input_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 3,
    projection_dim: int = 128,
    model_type: str = "supcon"
) -> nn.Module:
    if model_type == "supcon":
        return SupConProbe(input_dim, hidden_dim, output_dim, projection_dim)
    elif model_type == "supcon_dual":
        return SupConDualProbe(input_dim, hidden_dim, output_dim, projection_dim)
    elif model_type == "simple":
        return SimpleProbe(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


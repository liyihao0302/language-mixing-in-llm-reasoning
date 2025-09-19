# src/probe/test_supcon.py

import torch
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from probe.evaluate import evaluate_predictions, compute_decision_utility
from probe.model import get_model


def test_supcon_probe(
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df,
    checkpoint_path: str,
    input_dim: int,
    best_threshold,
    projection_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 64,
    use_metadata: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "supcon",
    log_to_mlflow: bool = True,
    return_probs: bool = False,
) -> tuple:
    """
    Run evaluation on held-out test set using a trained SupCon probe.

    Args:
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Ground truth labels
        test_df (pd.DataFrame): Test metadata (includes if_natural column)
        checkpoint_path (str): Path to trained model weights
        input_dim (int): Input dimension of the features
        projection_dim (int): Projection layer dim (default: 128)
        hidden_dim (int): Hidden layer dim (default: 256)
        batch_size (int): Batch size for DataLoader
        device (str): Device to run model on
        model_type (str): Probe architecture ("supcon" or "simple")
        log_to_mlflow (bool): Whether to log results to MLflow
        return_probs (bool): Whether to return softmax probabilities for threshold tuning

    Returns:
        Tuple[
            np.ndarray,  # Predicted labels
            np.ndarray,  # Ground truth labels
            np.ndarray,  # is_natural flags
            np.ndarray,  # Logits
            Optional[np.ndarray]  # Softmax probabilities
        ]
    """
    num_classes = int(np.max(y_test)) + 1
    is_natural = test_df["if_natural"].astype(bool).values

    # === 1. Load trained model ===
    model = get_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        projection_dim=projection_dim,
        model_type=model_type
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # === 2. Prepare test DataLoader ===
    test_tensor = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    test_loader = DataLoader(test_tensor, batch_size=batch_size)

    # === 3. Inference ===
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            _, logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.vstack(all_logits)
    all_probs = softmax(torch.tensor(all_logits), dim=1).numpy()
    if best_threshold is not None:
        tau_h, tau_help = best_threshold
        # vectorized thresholding:
        #   pred = 0 if p[0]>tau_h;
        #   else 2 if p[2]>tau_help;
        #   else 1
        all_preds = np.where(
            all_probs[:, 0] > tau_h, 0,
            np.where(all_probs[:, 2] > tau_help, 2, 1)
        )

    # === 4. Metrics and logging ===
    print("=== Test Set Evaluation ===")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    evaluate_predictions(all_labels, all_preds)

    utility = compute_decision_utility(all_labels, all_preds, is_natural)
    print(f"Decision Utility Score: {utility:.4f}")

    # === 5. MLflow logging ===
    if log_to_mlflow:
        mlflow.log_metric("test_macro_f1", macro_f1)
        mlflow.log_metric("test_decision_utility", utility)

        # Log confusion matrix as image
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Test Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

    if return_probs:
        return all_preds, all_labels, is_natural, all_logits, all_probs
    else:
        return all_preds, all_labels, is_natural, all_logits

## 📘 README Section: `test_supcon.py`

### Purpose
This module provides a utility for evaluating a trained supervised contrastive probe on a held-out test set. It measures both traditional classification performance (macro F1) and a custom decision utility metric based on whether code-switching was natural or forced. Optionally logs evaluation results and visualizations to MLflow.

### Key Features
- Load a trained probe model checkpoint
- Compute classification metrics and decision utility
- Support for both logits and softmax probability outputs
- Optional MLflow logging (metrics + confusion matrix)
- Designed to integrate seamlessly with `train_supcon.py`

---

## 🧾 API Documentation

### `test_supcon_probe(...)`

```python
def test_supcon_probe(
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    checkpoint_path: str,
    input_dim: int,
    projection_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 64,
    device: str = "cuda",
    model_type: str = "supcon",
    log_to_mlflow: bool = True,
    return_probs: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]
```

#### Args:
- **X_test** (`np.ndarray`): Feature matrix of test samples.
- **y_test** (`np.ndarray`): Integer labels (0=harmful, 1=neutral, 2=beneficial).
- **test_df** (`pd.DataFrame`): Metadata dataframe (must include `"if_natural"` column).
- **checkpoint_path** (`str`): Path to saved PyTorch model.
- **input_dim** (`int`): Dimensionality of test features.
- **projection_dim** (`int`, optional): Dim of contrastive projection head.
- **hidden_dim** (`int`, optional): Size of hidden MLP layers.
- **batch_size** (`int`, optional): Evaluation batch size.
- **device** (`str`, optional): Device to evaluate on (default: `"cuda"` if available).
- **model_type** (`str`, optional): Type of probe model ("supcon" or "simple").
- **log_to_mlflow** (`bool`, optional): Whether to log metrics/artifacts to MLflow.
- **return_probs** (`bool`, optional): If True, return softmax probabilities.

#### Returns:
- **y_pred** (`np.ndarray`): Predicted class labels.
- **y_true** (`np.ndarray`): Ground truth class labels.
- **is_natural** (`np.ndarray`): Boolean array from test_df.
- **logits** (`np.ndarray`): Raw logits output.
- **probs** (`np.ndarray`, optional): Softmax probabilities (if `return_probs=True`).

---

## 💡 Usage Example

```python
from probe.test_supcon import test_supcon_probe

# Test features and metadata
X_test, y_test = ...  # from train_supcon_probe return or PCA pipeline
test_df = ...         # contains "if_natural" column

# Path to trained model
ckpt_path = "checkpoints/supcon_probe_multilayer.pt"

# Evaluate on test set
y_pred, y_true, is_natural, logits, probs = test_supcon_probe(
    X_test=X_test,
    y_test=y_test,
    test_df=test_df,
    checkpoint_path=ckpt_path,
    input_dim=256,
    log_to_mlflow=True,
    return_probs=True
)

```

## 📄 README: SupCon Probe Training

### Overview

The `train_supcon_probe` function trains a supervised contrastive (SupCon) probe to predict whether a code-switch is harmful, neutral, or helpful. It uses pre-extracted LLM activations and associated metadata from multiple datasets (e.g., Math500, Gaokao MathQA, and Gaokao Cloze).

This component supports:
- Multi-layer activation fusion
- PCA-based dimensionality reduction (applied only to activations)
- Balanced batch sampling to address class imbalance
- Joint contrastive + cross-entropy optimization
- Utility-based evaluation (custom reward matrix)

---

### File Location

`src/probe/train_supcon.py`

---

### Input Requirements

- Pre-extracted activation files: `activations_*.h5` (for each dataset)
- Aligned metadata: `stats_*.csv` (with fields like `idx`, `problem_id`, `with_switch_score`, etc.)
- `dataset_src` is inferred automatically based on subdirectory naming conventions.

---

## 🔧 API: `train_supcon_probe`

```python
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
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, pd.DataFrame]]
```

### Parameters

| Name              | Type              | Description |
|-------------------|-------------------|-------------|
| `dataset_configs` | List[Tuple[str]]  | List of (subdir, csv_filename, h5_filename) for each dataset. |
| `layers`          | List[int]         | Layer indices to extract from activations. |
| `pca_dim`         | int               | Number of PCA components for dimensionality reduction. |
| `save_path`       | str               | Path to save trained model. |
| `input_dim`       | int               | Dimensionality of model input after PCA + metadata. |
| `hidden_dim`      | int               | Hidden layer size in probe. |
| `projection_dim`  | int               | Size of SupCon projection head. |
| `use_metadata`    | bool              | Whether to append metadata features. |
| `alpha`           | float             | Weight for contrastive loss. |
| `temperature`     | float             | Contrastive loss temperature. |
| `class_weights`   | List[float]       | Class weights for imbalanced CE loss. |
| `batch_size`      | int               | Batch size for training. |
| `num_epochs`      | int               | Number of training epochs. |
| `learning_rate`   | float             | Learning rate. |
| `val_ratio`       | float             | Fraction of data for validation. |
| `test_ratio`      | float             | Fraction of data for test. |
| `random_state`    | int               | Seed for reproducibility. |
| `device`          | str               | Device ("cuda" or "cpu"). |

---

### Returns

| Item            | Type                                | Description |
|------------------|-------------------------------------|-------------|
| `model`          | `nn.Module`                         | Trained PyTorch model |
| `val_logits`     | `np.ndarray` (N x 3)                | Raw logits on validation set |
| `val_labels`     | `np.ndarray`                        | True labels on validation set |
| `val_is_natural` | `np.ndarray` (bools)                | If sample had a natural code-switch |
| `(X_test, y_test, df_test)` | Tuple[np.ndarray, np.ndarray, pd.DataFrame] | Test features, labels, and metadata |

---

## 📘 Example Usage

```python
from probe.train_supcon import train_supcon_probe

dataset_configs = [
    ("outputs/math500_CH_1", "stats_math500.csv", "activations_math500.h5"),
    ("outputs/math500_CH_2", "stats_math500.csv", "activations_math500.h5"),
    ("outputs/gaokao_mathqa_CH", "stats_gaokao_mathqa.csv", "activations_gaokao_mathqa.h5"),
    ("outputs/gaokao_cloze_CH", "stats_gaokao_cloze.csv", "activations_gaokao_cloze.h5")
]

model, val_logits, val_labels, val_is_natural, (X_test, y_test, df_test) = train_supcon_probe(
    dataset_configs=dataset_configs,
    layers=[63, 47, 31, 15, 0],
    pca_dim=256,
    save_path="checkpoints/supcon_probe.pt"
)
```

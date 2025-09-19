

## 📘 README Section: `prepare_data.py`

This module handles data preparation for contrastive and classification probing on multilingual LLM activations. It supports:

- Loading and aligning stats and activations from multiple datasets
- Annotating each switch point as helpful / harmful / neutral
- Concatenating activations across multiple layers and attaching metadata
- Stratified group-aware train/val/test splits
- PCA-based dimensionality reduction applied only to activation features

### 📂 Supported Datasets

- `math500_CH_*`
- `gaokao_mathqa_CH`
- `gaokao_cloze_CH`

Each sample is associated with a dataset source (`dataset_src`) and a source question identifier (`problem_id`), ensuring that all switches from the same question are kept together during splitting.

---

## 🧩 API Documentation

### `load_all_raw_activations(...)`

```python
def load_all_raw_activations(
    dataset_configs: List[Tuple[str, str, str]],
    layers: Union[int, List[int]] = 63,
    use_metadata: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]
```

Loads activations and labels from multiple datasets.

- `dataset_configs`: List of tuples (subdir, stats.csv filename, h5 filename)
- `layers`: One or more activation layers to concatenate
- `use_metadata`: Whether to append heuristic metadata (e.g., entropy)

**Returns:**
- `X_all`: Combined feature matrix
- `y_all`: Combined class labels (0, 1, 2)
- `df_all`: Combined dataframe with `problem_id`, `dataset_src`, `label`, etc.

---

### `stratified_group_split_by_dataset(...)`

```python
def stratified_group_split_by_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Performs stratified train/val/test split by grouping on `(dataset_src, problem_id)`.

**Returns:**
- Boolean masks for `train_mask`, `val_mask`, `test_mask`

Also prints:
- Total count for each split
- Class distribution (harmful, neutral, helpful)

---

### `apply_pca_split(...)`

```python
def apply_pca_split(
    X: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]
```

Applies PCA only to the activation portion (excluding metadata). Fits PCA and normalization on the **training set only**, then applies to all splits.

**Returns:**
- Transformed `X_train`, `X_val`, `X_test`
- Fitted `PCA` and `StandardScaler`

---

## 🧪 Example Usage

```python
from probe.prepare_data import (
    load_all_raw_activations,
    stratified_group_split_by_dataset,
    apply_pca_split,
)

# 1. Load features and labels
dataset_configs = [
    ("outputs/math500_CH_1", "stats_math500.csv", "activations_math500.h5"),
    ("outputs/gaokao_mathqa_CH", "stats_gaokao_mathqa.csv", "activations_gaokao_mathqa.h5"),
    ("outputs/gaokao_cloze_CH", "stats_gaokao_cloze.csv", "activations_gaokao_cloze.h5"),
]
X_raw, y_raw, df_raw = load_all_raw_activations(dataset_configs, layers=[63, 47, 31, 15, 0])

# 2. Stratified group split
train_mask, val_mask, test_mask = stratified_group_split_by_dataset(df_raw)

# 3. Apply PCA
X_train, X_val, X_test, pca, scaler = apply_pca_split(
    X_raw, df_raw, train_mask, val_mask, test_mask, pca_dim=256
)
```

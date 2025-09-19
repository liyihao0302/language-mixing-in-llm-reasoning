# src/probe/prepare_data.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Union
from utils.load_activations import load_all_layer_activations
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5py

# Global config for metadata features
METADATA_COLUMNS = ["heuristic", "if_en_to_ch", "if_natural"]

def infer_dataset_source(subdir: str) -> str:
    if "math500" in subdir:
        return "math500"
    elif "gaokao_mathqa" in subdir:
        return "gaokao_mathqa"
    elif "gaokao_cloze" in subdir:
        return "gaokao_cloze"
    else:
        raise ValueError(f"Unknown dataset source from subdir: {subdir}")

def assign_label(delta_score: float) -> int:
    if delta_score > 0:
        return 2  # Helpful
    elif delta_score < 0:
        return 0  # Harmful
    else:
        return 1  # Neutral

def build_metadata_features(row: pd.Series) -> List[float]:
    return [float(row[col]) for col in METADATA_COLUMNS]

def load_all_raw_activations(
    dataset_configs: List[Tuple[str, str, str]],
    layers: Union[int, List[int]] = 63,
    use_metadata: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    all_X, all_y, all_df = [], [], []

    if isinstance(layers, int):
        layers = [layers]

    for subdir, csv_file, h5_file in dataset_configs:
        dataset_src = infer_dataset_source(subdir)
        csv_path = os.path.join(subdir, csv_file)
        h5_path = os.path.join(subdir, h5_file)

        df = pd.read_csv(csv_path)
        df = df[(df["with_switch_score"] >= 0) & (df["without_switch_score"] >= 0)].copy()
        df["delta_score"] = df["with_switch_score"] - df["without_switch_score"]
        df["label"] = df["delta_score"].apply(assign_label)
        df["dataset_src"] = dataset_src

        y = df["label"].values
        idx_array = df["idx"].astype(int).values

        
        
        features_by_layer = []
        with h5py.File(h5_path, "r") as f:
            for layer in layers:
                key = f"activations_{layer}"
                if key not in f:
                    raise ValueError(f"Missing {key} in {h5_path}")
                activations = f[key][:]
                
                if np.max(idx_array) >= len(activations):
                    raise IndexError(f"Index {np.max(idx_array)} out of bounds for layer {layer} in {h5_path}")
                features_by_layer.append(activations[idx_array])

        X = np.concatenate(features_by_layer, axis=1)

        if use_metadata:
            meta_features = df[METADATA_COLUMNS].astype(float).values
            X = np.concatenate([X, meta_features], axis=1)

        print(f"Loaded {len(df)} samples from {subdir} (source: {dataset_src})")
        all_X.append(X)
        all_y.append(y)
        all_df.append(df)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    df_all = pd.concat(all_df, ignore_index=True)


    # 1) grab the two token counts and delta_score
    w  = df_all["with_switch_token_count"].astype(float).values   # shape (N,)
    wo = df_all["without_switch_token_count"].astype(float).values
    delta = df_all["delta_score"].values

    # 2) compute relative change r = (w - wo)/wo, clamp to [-1,1]
    r = (w - wo) / wo
    r = np.clip(r, -1.0, 1.0)

    # 3) get preliminary soft probs for neutral cases
    p_help  = np.clip(-r, 0.0, 1.0)
    p_harm  = np.clip( r, 0.0, 1.0)
    p_neu   = 1.0 - p_help - p_harm

    # 4) stack into (N,3): [harm, neu, help]
    dist = np.vstack([p_harm, p_neu, p_help]).T  # shape (N,3)

    # 5) override with hard one‐hot whenever score_diff != 0
    helpful_mask = (delta > 0)
    harmful_mask = (delta < 0)

    dist[helpful_mask] = np.array([0.0, 0.0, 1.0])
    dist[harmful_mask] = np.array([1.0, 0.0, 0.0])
    # neutral_mask = delta == 0 already has the soft version

    return X_all, y_all, df_all, dist

def apply_pca_split(
    X: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]:
    num_meta_features = len(METADATA_COLUMNS)
    if num_meta_features >= X.shape[1]:
        raise ValueError("Metadata features exceed or equal input dimensionality.")

    # Separate activations and metadata
    X_act = X[:, :-num_meta_features]
    X_meta = X[:, -num_meta_features:]

    # Normalize and apply PCA on activation features
    scaler = StandardScaler().fit(X_act[train_idx])
    X_scaled = scaler.transform(X_act)
    pca = PCA(n_components=pca_dim).fit(X_scaled[train_idx])
    X_pca = pca.transform(X_scaled)

    # Reattach metadata
    X_final = np.concatenate([X_pca, X_meta], axis=1)

    return X_final[train_idx], X_final[val_idx], X_final[test_idx], pca, scaler

def stratified_group_split_by_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices of a dataframe into train, val, and test splits while grouping by (dataset_src, problem_id).

    Args:
        df (pd.DataFrame): Input dataframe with 'dataset_src', 'problem_id', and 'label' columns.
        val_ratio (float): Fraction of data to use for validation.
        test_ratio (float): Fraction of data to use for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple of boolean arrays (train_mask, val_mask, test_mask) over df rows.
    """
    required_columns = {"dataset_src", "problem_id", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataframe must contain columns: {required_columns}")

    # Group by dataset + problem
    group_keys = df[["dataset_src", "problem_id"]].drop_duplicates().copy()
    group_keys["group_id"] = range(len(group_keys))

    # Merge back group ID to dataframe
    df_merged = df.merge(group_keys, on=["dataset_src", "problem_id"], how="left")
    group_ids = df_merged["group_id"].unique()

    # First split into train+val and test
    train_val_ids, test_ids = train_test_split(
        group_ids, test_size=test_ratio, random_state=random_state
    )
    # Then split train_val into train and val
    val_fraction = val_ratio / (1 - test_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_fraction, random_state=random_state
    )

    # Create boolean masks for each split
    train_mask = df_merged["group_id"].isin(train_ids).values
    val_mask = df_merged["group_id"].isin(val_ids).values
    test_mask = df_merged["group_id"].isin(test_ids).values

    # Print useful stats
    def print_stats(mask: np.ndarray, name: str):
        labels = df.loc[mask, "label"]
        counts = labels.value_counts().sort_index()
        print(f"{name} set size: {len(labels)}")
        for label in sorted(counts.index):
            print(f"  Class {label}: {counts[label]}")
        print()

    print_stats(train_mask, "Train")
    print_stats(val_mask, "Validation")
    print_stats(test_mask, "Test")

    return train_mask, val_mask, test_mask

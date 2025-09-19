import h5py
import numpy as np
from typing import Dict

def load_all_layer_activations(h5_path: str) -> Dict[int, np.ndarray]:
    """
    Load all available activations from the .h5 file.

    Returns:
        Dict[int, np.ndarray]: Mapping from layer number to activation matrix.
    """
    activations = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if key.startswith("activations_"):
                layer = int(key.split("_")[-1])
                activations[layer] = f[key][:]
    return activations
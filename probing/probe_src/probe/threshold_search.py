# probe/threshold_search.py
import numpy as np
from itertools import product
import torch.nn.functional as F
from typing import List, Tuple

def compute_decision_utility(y_true, y_pred, is_natural):
    """
    Simplified utility:
    +1 for correct class
    -1 for harmful interventions
    0 otherwise
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_natural = np.asarray(is_natural)

    scores = []
    for yt, yp, nat in zip(y_true, y_pred, is_natural):
        if nat and yt == yp and yt == 0:
            scores.append(1)
        elif not nat and yt == yp and yt == 2:
            scores.append(1)
        elif nat and yt == 2 and yp == 0:  
            scores.append(-1)
        elif not nat and yt == 0 and yp == 2: 
            scores.append(-1)
        else:
            scores.append(0)
    return np.mean(scores)

def threshold_search_utility(
    probs: np.ndarray,
    y_true: np.ndarray,
    is_natural: np.ndarray,
    tau_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.05,
    verbose: bool = True
) -> Tuple[Tuple[float, float], float, np.ndarray]:
    tau_values = np.arange(tau_range[0], tau_range[1] + step, step)
    best_util = -np.inf
    best_thresh = (0.5, 0.5)
    y_pred_best = None

    for tau_h, tau_help in product(tau_values, repeat=2):
        y_pred = []
        for p in probs:
            if p[0] > tau_h:
                y_pred.append(0)
            elif p[2] > tau_help:
                y_pred.append(2)
            else:
                y_pred.append(1)

        util = compute_decision_utility(y_true, y_pred, is_natural)

        if util > best_util:
            best_util = util
            best_thresh = (tau_h, tau_help)
            y_pred_best = y_pred

        if verbose:
            print(f"τ_h={tau_h:.2f}, τ_help={tau_help:.2f} → Utility Score = {util:.4f}")

    print(f"\nBest thresholds: τ_h={best_thresh[0]:.2f}, τ_help={best_thresh[1]:.2f}, Utility = {best_util:.4f}")
    return best_thresh, best_util, np.array(y_pred_best)

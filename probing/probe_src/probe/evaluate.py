# src/probe/evaluate.py
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> dict:
    """
    Compute macro F1, confusion matrix, and full classification report.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        verbose (bool): If True, print the report and matrix.

    Returns:
        dict: Dictionary with 'macro_f1', 'report', and 'confusion_matrix'.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=3, output_dict=False)
    matrix = confusion_matrix(y_true, y_pred)

    if verbose:
        print("Confusion Matrix:\n", matrix)
        print("\nClassification Report:")
        print(report)
        print(f"\nMacro F1 Score: {macro_f1:.4f}")

    
    return {
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": matrix
    }


def compute_decision_utility(y_true, y_pred, is_natural):
    """
    Computes custom decision utility for 3-class prediction with case-specific cost.

    Args:
        y_true (array-like): Ground truth labels (0 = harmful, 1 = neutral, 2 = beneficial)
        y_pred (array-like): Predicted labels
        is_natural (array-like): Boolean array, True = natural switch case (Case 1), False = no switch (Case 2)

    Returns:
        float: Average decision utility score over all samples
    """
    # Case 1: natural switch
    #       Harmful   Neutral   Beneficial
    # GT: 
    #  H      +1        0         0
    #  N      0         0          0
    #  B      -1        0         0
    utility_case1 = np.array([
        [ 1, 0, 0],
        [0,  0,  0],
        [-1,  0, 0]
    ])

    # Case 2: no natural switch
    #       Harmful   Neutral   Beneficial
    # GT:
    #  H       0        0         -1
    #  N      0         0         0
    #  B      0        0         +1
    utility_case2 = np.array([
        [ 0,  0, -1],
        [ 0,  0, 0],
        [0, 0,  +1]
    ])

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_natural = np.asarray(is_natural)

    scores = []
    for yt, yp, natural in zip(y_true, y_pred, is_natural):
        if yt not in [0, 1, 2] or yp not in [0, 1, 2]:
            scores.append(0)
            continue
        matrix = utility_case1 if natural else utility_case2
        scores.append(matrix[yt, yp])

    return np.mean(scores)




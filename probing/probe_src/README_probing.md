# Probing Code-Switching Dynamics in LLM Activations

This repository provides tools for probing language model activations at code-switching points (e.g., between English and Chinese) in math word problem datasets like MATH500. It supports both traditional classification probes and supervised contrastive learning to identify helpful, neutral, or harmful code-switching behavior.

---

## 📁 Directory Structure

```
probe_src/
├── checkpoints/                     # Saved model checkpoints
├── outputs/                         # Output activations and stats from data generation
├── probe/                           # Probing models, training logic, evaluation tools
│   ├── model.py                     # Model architectures (MLP, SupCon probe)
│   ├── train.py                     # Standard probe training
│   ├── train_supcon.py              # SupCon-based training (hybrid loss)
│   ├── evaluate.py                  # Evaluation metrics (F1, confusion matrix)
│   ├── test.py                      # Test loader for standard probe
│   ├── test_supcon.py               # Test loader for SupCon probe
│   ├── contrastive_loss.py          # Supervised Contrastive Loss
│   └── prepare_data.py              # Data loading, filtering, stratified splitting
├── utils/                           # Helper functions
│   └── load_activations.py          # Read H5 activations aligned with CSV metadata
├── Probe_Training_and_Evaluation.ipynb         # Notebook for training + testing standard probes
├── SupCon_Prob_Train_Eval.ipynb                # Notebook for SupCon probe training + eval
```

---

## 🧠 Probe Goals

The core task is to predict whether a code-switching action (natural or synthetic) at a specific location in a generated response will be:
- ✅ Beneficial (improves answer correctness)
- ➖ Neutral (no change in answer)
- ❌ Harmful (worsens correctness)

---

## 🧹 Features and Inputs

- **Activations:** Layer-wise embeddings from LLM (e.g., Qwen) at switch positions.
- **Metadata:** Heuristic entropy, switch direction, etc.
- **Labels:** Derived from delta in model correctness pre/post code-switch.

---

## 🛠 Usage

### 1. Train a Standard MLP Probe

```python
from probe.train import train_probe

train_probe(
    input_dim=5124,
    data_root="outputs",
    subdirs=["math500_CH_1", "math500_CH_2"],
    layer=63,
    save_path="checkpoints/probe.pt"
)
```

### 2. Train a SupCon Hybrid Probe

```python
from probe.train_supcon import train_supcon_probe

train_supcon_probe(
    input_dim=256,
    subdirs=["math500_CH_1", "math500_CH_2"],
    layer=63,
    projection_dim=128,
    pca_dim=256,
    save_path="checkpoints/supcon_probe.pt"
)
```

### 3. Test on Held-Out Set

```python
from probe.test import test_probe
from probe.test_supcon import test_supcon_probe

# For standard
test_probe(checkpoint_path="checkpoints/probe.pt", ...)

# For SupCon
test_supcon_probe(checkpoint_path="checkpoints/supcon_probe.pt", ...)
```

---

## 📊 Supported Experiments

- Ablation on layer depth
- Use of metadata
- PCA vs raw activations
- MLP vs SupCon hybrid objectives
- Stratified group split (by problem ID)

---


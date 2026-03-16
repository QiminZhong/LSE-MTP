
# Toward Consistent World Models with LSE-MTP

This is the official repository for the paper: **"Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement"**.

This project provides a unified framework to train and evaluate Large Language Models (LLMs) on graph-based navigation tasks. It introduces **LSE-MTP**, a method that anchors multi-token predictions to ground-truth latent trajectories to prevent structural hallucinations and foster coherent internal world models.

---

## 1. Installation

Environment configuration is provided via `spec-file.txt`. You can create a compatible Conda environment using:

```bash
conda create --name lse_mtp --file spec-file.txt
conda activate lse_mtp
```

---

## 2. Dataset Generation (`generate_dataset.py`)

This script generates synthetic graph environments and samples paths. Paths are automatically converted into an **Incremental Representation** (e.g., `[Source, Target, inc_1, inc_2, ...]`). It saves binary data for training, a `.txt` file for evaluation, and a `.graphml` file for topology verification.

### Parameters:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--type` | `str` | `usg` | Graph topology: `er` (Erdős–Rényi) or `usg` (Urban Street Graph). |
| `--nodes` | `int` | `100` | Number of nodes ($N$) in the graph. |
| `--edge_p` | `float` | `0.3` | Edge probability (ER) or mesh density (USG). |
| `--train_ratio` | `float` | `0.5` | Fraction of reachable node pairs assigned to the training set. |
| `--k_paths` | `int` | `3` | Top-K shortest paths to sample for each node pair. |
| `--detour_prob` | `float` | `0.3` | Probability of generating a detour path (robustness). |
| `--recovery_prob`| `float` | `0.3` | Probability of generating a recovery path (error correction). |

**Example:**
```bash
python generate_dataset.py --type er --nodes 100 --edge_p 0.04
```

```bash
python generate_dataset.py --type usg --nodes 100 --edge_p 0.3
```

---

## 3. Model Training (`train.py`)

The unified training script supports **Next-Token Prediction (NTP)**, **Multi-Token Prediction (MTP)**, and **LSE-MTP**. It uses Distributed Data Parallel (DDP) and `torch.compile` for high-performance training.

### Parameters:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `er` | Dataset type used during generation (`er` or `usg`). |
| `--num_nodes` | `int` | `100` | Number of nodes in the generated dataset. |
| `--method` | `str` | `lse` | Training method: `standard` (NTP/MTP) or `lse` (LSE-MTP). |
| `--n_tokens` | `int` | `1` | Total prediction horizon $K$ (e.g., 1 for NTP, 4 for MTP). |
| `--lambda_l` | `float` | `0.1` | Weight for Latent Consistency Loss ($\lambda_l$). |
| `--lambda_s` | `float` | `0.1` | Weight for Semantic Anchoring Loss ($\lambda_s$). |
| `--n_layer` | `int` | `6` | Number of Transformer layers. |
| `--batch_size` | `int` | `1024` | Samples per batch (automatically aligned to block size). |
| `--lr` | `float` | `5e-4` | Peak learning rate with Cosine decay. |

**Example (LSE-MTP with K=4):**
```bash
python train.py --method lse --n_tokens 4 --dataset er --num_nodes 100
```

---

## 4. Evaluation Suite

All evaluation scripts support two checkpoint loading modes:
1.  **Auto Mode**: Automatically constructs the path based on training parameters (e.g., `--method`, `--n_tokens`).
2.  **Manual Mode**: Loads a specific `.pt` file via the `--checkpoint` argument (overrides Auto Mode).

### A. Pathfinding Accuracy (`test_accuracy.py`)
Measures the success rate of the model in generating valid graph-consistent paths from a starting node to a target goal.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `er` | Dataset type (`er` or `usg`). |
| `--num_nodes` | `int` | `100` | Number of nodes in the graph. |
| `--method` | `str` | `lse` | Training method (`standard` or `lse`). |
| `--n_tokens` | `int` | `1` | Prediction horizon $K$ used during training. |
| `--ckpt_iter` | `str` | `final` | Checkpoint iteration to load (`final` or e.g., `5000`). |
| `--checkpoint` | `str` | `''` | Manual path to a `.pt` file (highest priority). |
| `--batch_size` | `int` | `100` | Number of parallel inference tasks. |
| `--temperature` | `float` | `0.8` | Softmax temperature (lower is more deterministic). |
| `--device` | `str` | `cuda` | Hardware device (`cuda` or `cpu`). |

**Example:**
```bash
python test_accuracy.py --method lse --n_tokens 4 --temperature 0.8
```

---

### B. Representation Alignment (`evaluation/eval_representation_alignment.py`)
Calculates **Structure Gain** (Table 1) by measuring the cosine similarity between hidden states that share the same future target but reached it via different immediate actions.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--k` | `int` | `3` | Metric Horizon: the future step to evaluate alignment. |
| `--max_samples`| `int` | `1000` | Number of paths to sample from the validation set. |
| `--seed` | `int` | `42` | Random seed for reproducible sampling. |
| `--dataset` | `str` | `usg` | Dataset type used for state extraction. |
| `--num_nodes` | `int` | `100` | Graph size. |
| `--method` | `str` | `standard`| Model method (`standard` or `lse`). |
| `--n_tokens` | `int` | `1` | Horizon $K$. |

**Example:**
```bash
python evaluation/eval_representation_alignment.py --method lse --n_tokens 4 --k 3
```

---

### C. Belief Compression (`evaluation/eval_belief_compression.py`)
Computes **Belief Similarity** and **Goal Distinction** (Table 2) using controlled group trials to evaluate how the latent space abstracts different paths to the same goal.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num_trials` | `int` | `1000` | Number of trials to run for each control group. |
| `--seed` | `int` | `42` | Seed for reproducible test group generation. |
| `--dataset` | `str` | `usg` | Graph type. |
| `--num_nodes` | `int` | `100` | Graph size. |
| `--method` | `str` | `standard`| Model method. |
| `--n_tokens` | `int` | `1` | Horizon $K$. |
| `--ckpt_iter` | `str` | `final` | Checkpoint to evaluate. |

**Example:**
```bash
python evaluation/eval_belief_compression.py --method lse --n_tokens 4 --num_trials 1000
```

---

### D. Structural Hallucinations (`evaluation/eval_structural_hallucinations.py`)
Quantifies the **Illegal Shortcut Probability (ISP)** (Table 3) to detect if the model hallucinates direct transitions between nodes that share a future goal but are not physically connected.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--max_samples`| `int` | `5000` | Number of $(A, B, G)$ structural candidates to mine. |
| `--seed` | `int` | `42` | Seed to ensure the same candidates are used across runs. |
| `--dataset` | `str` | `usg` | Graph type. |
| `--num_nodes` | `int` | `100` | Graph size. |
| `--method` | `str` | `lse` | Model method. |
| `--n_tokens` | `int` | `4` | Horizon $K$. |
| `--device` | `str` | `cuda` | Evaluation hardware. |

**Example:**
```bash
python evaluation/eval_structural_hallucinations.py --method standard --n_tokens 4 --max_samples 5000
```

---

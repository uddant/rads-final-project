"""Central configuration for the federated Count Sketch experiment.

Every major hyperparameter lives in this file so experiments are easy to
inspect and modify.  The training code imports these constants instead of
hard-coding values, which makes the project easier to reproduce.

DEVICE: automatically chooses GPU when PyTorch can see CUDA, otherwise CPU.
SEED: random seed used for Python, NumPy, and PyTorch reproducibility.
NUM_CLIENTS: number of simulated federated clients.
CLIENTS_PER_ROUND: number of clients sampled in each communication round.
LOCAL_EPOCHS: number of local passes over a client's private data per round.
BATCH_SIZE: mini-batch size for local client training and test evaluation.
LEARNING_RATE: SGD learning rate used by each client.
NUM_ROUNDS: number of federated communication rounds.
SHARDS_PER_CLIENT: number of label-sorted shards assigned to each client;
    with MNIST this means clients usually see only a few digit classes.
SKETCH_ROWS: number of independent Count Sketch hash/sign rows.
SKETCH_COMPRESSION_RATIOS: target compression factors for Count Sketch runs.
"""

from pathlib import Path

import torch

# Hardware selection.  All other files use this torch.device object rather
# than hard-coding .cuda() or .cpu(), so the same code works on GPU and CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_ROUNDS = 50
SHARDS_PER_CLIENT = 2

EMNIST_SPLIT = "balanced"
NUM_CLASSES = 47

SKETCH_ROWS = 10
SKETCH_COMPRESSION_RATIOS = [2, 5, 10, 50, 100, 200, 1000]

# Project paths.  Using pathlib keeps path handling readable and portable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

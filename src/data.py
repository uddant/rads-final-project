"""MNIST loading and non-IID client partitioning utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from . import config


def load_mnist() -> tuple[datasets.EMNIST, datasets.EMNIST]:
    """Download and return MNIST train/test datasets.

    Returns:
        A pair ``(train_dataset, test_dataset)``.  Each item is a normalized
        tensor image and an integer label.  We use torchvision's EMNIST class so
        the project stays compact and reproducible.
    """
    # EMNIST pixels are originally integers in [0, 255].  ToTensor converts them
    # to floats in [0, 1].  The normalization values below are the standard
    # EMNIST mean and standard deviation used in many PyTorch examples.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(0, 2, 1)),  # fix EMNIST transpose
        transforms.Normalize((0.1751,), (0.3332,)),        # EMNIST balanced stats
    ])
    
    train_dataset = datasets.EMNIST(
        root=str(config.DATA_DIR), split=config.EMNIST_SPLIT,
        train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        root=str(config.DATA_DIR), split=config.EMNIST_SPLIT,
        train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def get_emnist_label_map(split: str) -> Dict[int, str]:
    """Return a mapping from integer class index to human-readable character.

    Args:
        split: One of the torchvision EMNIST split names: ``"balanced"``,
            ``"byclass"``, ``"bymerge"``, ``"letters"``, ``"digits"``,
            ``"mnist"``.

    Returns:
        Dict mapping each class index (0-based, as torchvision produces them)
        to a display string such as ``'A'``, ``'n'``, or ``'7'``.

    EMNIST label ordering (torchvision remaps all splits to start at 0):
        - ``digits`` / ``mnist``: 0–9 → '0'–'9'  (identical to plain MNIST)
        - ``letters``: 0–25 → 'a'–'z'  (case-insensitive; torchvision subtracts 1
          from the raw 1-based labels)
        - ``balanced``: 0–9 digits, 10–35 A–Z, 36–46 visually-distinct lowercase
          letters (a b d e f g h n q r t)
        - ``byclass`` / ``bymerge``: 0–9 digits, 10–35 A–Z, 36–61 a–z
    """
    digits = {i: str(i) for i in range(10)}
    upper = {10 + i: chr(ord("A") + i) for i in range(26)}

    if split in ("digits", "mnist"):
        return digits

    if split == "letters":
        return {i: chr(ord("a") + i) for i in range(26)}

    if split == "balanced":
        # These 11 lowercase letters were kept because they are visually
        # distinct enough from their uppercase counterparts in the dataset.
        balanced_lower_chars = list("abdefghnqrt")
        lower = {36 + i: c for i, c in enumerate(balanced_lower_chars)}
        return {**digits, **upper, **lower}

    if split in ("byclass", "bymerge"):
        lower = {36 + i: chr(ord("a") + i) for i in range(26)}
        return {**digits, **upper, **lower}

    raise ValueError(f"Unknown EMNIST split: {split!r}")

def partition_non_iid(
    train_dataset: datasets.MNIST,
    num_clients: int,
    shards_per_client: int,
) -> Dict[int, Subset]:
    """Create a classic pathological non-IID EMNIST split.

    Args:
        train_dataset: The EMNIST training dataset.
        num_clients: Number of federated clients to simulate.
        shards_per_client: Number of label-sorted shards assigned to each
            client.

    Returns:
        A dictionary mapping ``client_id`` to a ``torch.utils.data.Subset``.

    Why this split matters:
        In IID training, every client would receive a miniature version of the
        full MNIST distribution.  Real federated learning is rarely IID: one
        phone might contain many photos of a person's favorite digits, while
        another has different patterns.  Sorting by label and assigning a few
        shards per client exaggerates this statistical heterogeneity, making
        the problem harder and therefore more informative for FL experiments.
    """
    num_shards = num_clients * shards_per_client
    num_samples = len(train_dataset)
    shard_size = num_samples // num_shards

    # torchvision stores EMNIST labels in ``targets``.  We sort indices by label
    # so each shard is dominated by one digit class.
    labels = np.asarray(train_dataset.targets)
    sorted_indices = np.argsort(labels)

    # Drop a small remainder if the dataset size is not exactly divisible.  For
    # MNIST with 100 clients and 2 shards/client this divides evenly: 60,000 / 200.
    usable = shard_size * num_shards
    sorted_indices = sorted_indices[:usable]

    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size]
        for i in range(num_shards)
    ]

    rng = np.random.RandomState(config.SEED)
    shard_order = rng.permutation(num_shards)

    client_subsets: Dict[int, Subset] = {}
    for client_id in range(num_clients):
        chosen = shard_order[
            client_id * shards_per_client : (client_id + 1) * shards_per_client
        ]
        client_indices = np.concatenate([shards[shard_id] for shard_id in chosen])
        client_subsets[client_id] = Subset(train_dataset, client_indices.tolist())

    return client_subsets

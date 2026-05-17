"""Federated training loop for FedAvg and Count Sketch experiments."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .client import clone_state_dict, compute_update_vector, local_train
from .data import load_dataset, partition_non_iid
from .evaluate import evaluate_accuracy
from .model import SmallCNN
from .server import apply_update_vector, count_parameters, fedavg_aggregate, sketch_aggregate
from .sketch import CountSketch


def _make_client_loaders(cfg: ModuleType) -> tuple[dict[int, DataLoader], DataLoader]:
    """Load dataset, split clients, and build DataLoaders.

    Args:
        cfg: Configuration module containing experiment constants.

    Returns:
        ``(client_loaders, test_loader)``.
    """
    train_dataset, test_dataset = load_dataset()
    client_subsets = partition_non_iid(
        train_dataset, cfg.NUM_CLIENTS, cfg.SHARDS_PER_CLIENT
    )

    # Each client loader shuffles its local examples.  Federated clients train
    # independently, so this approximates the random order each device would see.
    client_loaders = {
        client_id: DataLoader(
            subset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        for client_id, subset in client_subsets.items()
    }

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )
    return client_loaders, test_loader


def _target_width(vector_dim: int, rows: int, compression_ratio: float) -> int:
    """Compute sketch width from a desired compression ratio.

    Args:
        vector_dim: Original update-vector length.
        rows: Number of Count Sketch rows.
        compression_ratio: Desired ``vector_dim / (rows * width)``.

    Returns:
        Integer sketch width, at least 1.
    """
    return max(1, int(round(vector_dim / (rows * compression_ratio))))


def _bits_for_method(vector_dim: int, cfg: ModuleType, width: int | None) -> int:
    """Estimate transmitted bits per selected client per round.

    Args:
        vector_dim: Number of full update coordinates.
        cfg: Configuration module.
        width: Sketch width, or ``None`` for FedAvg.

    Returns:
        Total float32 payload bits for all selected clients in one round.
    """
    floats_per_client = vector_dim if width is None else cfg.SKETCH_ROWS * width
    return int(cfg.CLIENTS_PER_ROUND * floats_per_client * 32)


def federated_train(
    method: str,
    sketch_compression_ratio: float | None = None,
    config: ModuleType = config,
) -> Dict[str, Any]:
    """Run one full federated learning experiment.

    Args:
        method: Either ``"fedavg"`` or ``"countsketch"``.
        sketch_compression_ratio: Target compression ratio for Count Sketch.
            Ignored for FedAvg.
        config: Module containing hyperparameters.

    Returns:
        Dictionary with accuracy history, communication cost, final model state,
        and metadata.  The same dictionary is saved to ``results/`` as a ``.pt``
        file so plotting can be done later without retraining.
    """
    if method not in {"fedavg", "countsketch"}:
        raise ValueError("method must be 'fedavg' or 'countsketch'.")
    if method == "countsketch" and sketch_compression_ratio is None:
        raise ValueError("Count Sketch requires sketch_compression_ratio.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client_loaders, test_loader = _make_client_loaders(config)
    global_model = SmallCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    vector_dim = count_parameters(global_model)

    sketcher = None
    width = None
    if method == "countsketch":
        width = _target_width(vector_dim, config.SKETCH_ROWS, float(sketch_compression_ratio))
        sketcher = CountSketch(
            num_rows=config.SKETCH_ROWS,
            num_cols=width,
            vector_dim=vector_dim,
            seed=config.SEED,
        )

    # Evaluate initially and then after each round.  Keeping every round makes
    # the result files more useful for later analysis, and evaluation is cheap
    # relative to many client training steps.
    accuracy_history: list[float] = [
        evaluate_accuracy(global_model, test_loader, config.DEVICE)
    ]
    bits_per_round = _bits_for_method(vector_dim, config, width)

    rng = random.Random(config.SEED)
    round_iter = tqdm(range(1, config.NUM_ROUNDS + 1), desc=f"Training {method}")

    for round_idx in round_iter:
        selected_clients = rng.sample(range(config.NUM_CLIENTS), config.CLIENTS_PER_ROUND)
        old_global_state = clone_state_dict(global_model.state_dict())

        update_vectors: list[torch.Tensor] = []
        sketches: list[torch.Tensor] = []

        for client_id in selected_clients:
            # Every client starts from the same global model for this round.
            client_model = copy.deepcopy(global_model)
            local_train(
                client_model,
                client_loaders[client_id],
                lr=config.LEARNING_RATE,
                epochs=config.LOCAL_EPOCHS,
                device=config.DEVICE,
            )

            update = compute_update_vector(old_global_state, client_model.state_dict())
            update = update.to(config.DEVICE)

            if method == "fedavg":
                update_vectors.append(update)
            else:
                assert sketcher is not None
                sketches.append(sketcher.sketch(update))

        if method == "fedavg":
            averaged_update = fedavg_aggregate(update_vectors)
        else:
            assert sketcher is not None
            averaged_update = sketch_aggregate(sketches, sketcher)

        apply_update_vector(global_model, averaged_update)
        accuracy = evaluate_accuracy(global_model, test_loader, config.DEVICE)
        accuracy_history.append(accuracy)
        round_iter.set_postfix(acc=f"{accuracy:.3f}")

    result: Dict[str, Any] = {
        "method": method,
        "compression_ratio": 1 if method == "fedavg" else sketch_compression_ratio,
        "actual_compression_ratio": 1 if method == "fedavg" else vector_dim / (config.SKETCH_ROWS * width),
        "sketch_rows": None if method == "fedavg" else config.SKETCH_ROWS,
        "sketch_width": width,
        "accuracy_history": accuracy_history,
        "bits_per_round": bits_per_round,
        "final_model_state_dict": {k: v.cpu() for k, v in global_model.state_dict().items()},
    }

    suffix = "baseline" if method == "fedavg" else f"ratio_{sketch_compression_ratio}"
    save_path = config.RESULTS_DIR / f"{method}_{suffix}.pt"
    torch.save(result, save_path)
    print(f"Saved {method} results to {save_path}")
    return result

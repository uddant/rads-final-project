"""Regenerate the three plots from saved experiment results without re-training."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import config
from src.data import load_mnist, get_emnist_label_map
from src.evaluate import capture_single_round_gradient
from src.model import SmallCNN
from src.server import count_parameters
from src.sketch import CountSketch
from plots.plotting import (
    plot_accuracy_vs_compression,
    plot_gradient_recovery,
    plot_prediction_grid,
)


def _result_path(method: str, ratio: int | None = None) -> Path:
    """Return the saved-result path for a training run."""
    if method == "fedavg":
        return config.RESULTS_DIR / "fedavg_baseline.pt"
    return config.RESULTS_DIR / f"countsketch_ratio_{ratio}.pt"


def _load_model_from_result(result: dict) -> SmallCNN:
    """Rebuild a SmallCNN and load a saved result state dict."""
    model = SmallCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(result["final_model_state_dict"])
    model.eval()
    return model


def main() -> None:
    """Load results and regenerate all three plots."""
    print("Loading saved results...")
    all_results = {}

    # Load FedAvg baseline
    fedavg_path = _result_path("fedavg")
    if not fedavg_path.exists():
        raise FileNotFoundError(f"FedAvg result not found at {fedavg_path}")
    fedavg_result = torch.load(fedavg_path, weights_only=False)
    all_results[1] = fedavg_result

    # Load Count Sketch results
    for ratio in config.SKETCH_COMPRESSION_RATIOS:
        cs_path = _result_path("countsketch", ratio)
        if cs_path.exists():
            all_results[ratio] = torch.load(cs_path, weights_only=False)
            print(f"  Loaded CS@{ratio}")
        else:
            print(f"  Warning: CS@{ratio} not found, skipping")

    # Plot 1: Accuracy vs Compression
    print("Generating accuracy_vs_compression.png...")
    plot_accuracy_vs_compression(
        all_results,
        config.FIGURES_DIR / "accuracy_vs_compression.png",
    )

    # Plot 2: Prediction Grid
    print("Generating prediction_grid.png...")
    _, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    models_for_grid = {"FedAvg": _load_model_from_result(fedavg_result)}
    for ratio in config.SKETCH_COMPRESSION_RATIOS:
        if ratio in all_results:
            models_for_grid[f"CS@{ratio}"] = _load_model_from_result(all_results[ratio])
    label_map = get_emnist_label_map(config.EMNIST_SPLIT)
    plot_prediction_grid(
        models_for_grid,
        test_loader,
        config.FIGURES_DIR / "prediction_grid.png",
        label_map=label_map,
    )

    # Plot 3: Gradient Recovery
    print("Generating gradient_recovery.png...")
    diagnostic_model = _load_model_from_result(fedavg_result)
    small_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    true_gradient = capture_single_round_gradient(diagnostic_model, small_loader, config.DEVICE)

    vector_dim = count_parameters(diagnostic_model)
    sketch_estimates = {}
    for ratio in config.SKETCH_COMPRESSION_RATIOS:
        width = max(1, int(round(vector_dim / (config.SKETCH_ROWS * ratio))))
        sketcher = CountSketch(
            num_rows=config.SKETCH_ROWS,
            num_cols=width,
            vector_dim=vector_dim,
            seed=config.SEED + 999,
        )
        sketch_estimates[ratio] = sketcher.unsketch(sketcher.sketch(true_gradient))

    plot_gradient_recovery(
        true_gradient,
        sketch_estimates,
        config.FIGURES_DIR / "gradient_recovery.png",
    )

    print(f"All plots saved to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()

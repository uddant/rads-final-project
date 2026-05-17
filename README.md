# Federated Learning with Count Sketch Compression on MNIST

This project implements a small, readable federated learning experiment on EMNIST. It compares standard uncompressed **FedAvg** against **Count Sketch-compressed client updates** at several compression ratios.

## Project layout

```text
fedlearn_countsketch/
├── run_experiment.py
├── src/                  # data, model, sketch, client/server, trainer, evaluation
├── plots/                # three required plotting functions
├── results/              # generated .pt result files
├── figures/              # generated .png figures
└── notebooks/            # standalone Count Sketch demo notebook
(you might need to manually create results and figures if they got gitignored)
```

## Installation

Create a clean Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The project uses only PyTorch, torchvision, NumPy, matplotlib, seaborn, tqdm, and Jupyter.

## Run the full experiment

From the project root:

```bash
python run_experiment.py
```

The script will:

1. Detect GPU if available, otherwise use CPU.
2. Train one FedAvg baseline.
3. Train Count Sketch runs for compression ratios, e.g. `[2, 5, 10, 20, 50, 100]`.
4. Save result files in `results/`.
5. Save figures in `figures/`:
   - `accuracy_vs_compression.png`
   - `prediction_grid.png`
   - `gradient_recovery.png`

## Notebook

Open the standalone Count Sketch tutorial with:

```bash
jupyter notebook notebooks/count_sketch_demo.ipynb
```

The notebook uses synthetic vectors only. It demonstrates Count Sketch intuition, recovery behavior, linearity, and how error changes with sketch size.

## Runtime notes

The default configuration uses 100 clients, 10 clients per round, 50 communication rounds, and one local epoch per selected client. If runtime is too high on your machine, reduce `NUM_ROUNDS` in `src/config.py` first

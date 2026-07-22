# Federated Learning under Count Sketch Compression

**An empirical study of how far a randomized sublinear-space algorithm can be pushed inside a real training loop.**

This repository trains a CNN on EMNIST across 100 simulated federated clients and asks a single question: *if every client compresses its model update with a Count Sketch before uploading it, how much bandwidth can we save before learning breaks?*

The short answer, measured over 50 communication rounds against an uncompressed FedAvg baseline:

| Compression | Final accuracy | Retained vs. FedAvg | Uplink per round |
|---|---|---|---|
| **1× (FedAvg)** | **0.420** | 100% | 7.57 MB |
| 10× | 0.328 | 78% | 0.76 MB |
| 100× | 0.175 | 42% | 0.08 MB |
| 1000× | 0.021 | 5% (= random guessing) | 0.008 MB |

Compression is not free even at 2×, degrades roughly *linearly in the logarithm* of the compression ratio, and then falls off a cliff into a degenerate classifier. The interesting part is not that it degrades — it is *where* and *why* it degrades, which is what the three figures below are for.

---

## 1. What is federated learning?

In conventional training, all data is pooled on one machine. Federated learning inverts that: the data never moves.

A central server holds a **global model**. Each round it:

1. **Broadcasts** the current global weights to a sample of participating clients (phones, hospitals, browsers).
2. Each selected client **trains locally** on its own private data for a few steps and computes an **update** — the difference between its post-training weights and the weights it was given.
3. Clients **upload only that update**. Raw data stays on the device.
4. The server **averages** the received updates and applies the average to the global model.

That loop is *FedAvg*, the baseline in this repository. Its defining practical problem is **communication**. Local computation is nearly free — a phone charging overnight has idle cycles — but every round every participant must ship a vector as large as the entire model over a consumer uplink. With the model used here, one round of ten clients moves 7.57 MB, and fifty rounds move roughly 379 MB, to train a model that fits comfortably in 750 KB. Compute scales; uplinks do not.

This makes federated learning an unusually honest testbed for compression: the quantity being compressed (a gradient-like update vector) is high-dimensional, noisy, and — crucially — only ever consumed *in aggregate*. Nobody needs any individual client's update. The server only needs their mean. That structural fact is exactly the opening a sketching algorithm exploits.

---

## 2. Count Sketch, at a high level

A **Count Sketch** is a randomized data structure that stores a summary of a $D$-dimensional vector in a $d \times w$ table where $d \cdot w \ll D$. It is a *linear sketch*: the summary is a fixed random linear map applied to the vector.

**Construction.** Fix $d$ independent pairs of random functions. Each row $r$ gets a hash $h_r$ that sends every coordinate index to one of $w$ buckets, and a sign $s_r \in \{-1, +1\}$ assigned to every coordinate. To sketch a vector $x$, each coordinate is added into its bucket with its sign:

$$\text{table}[r, h_r(j)] \mathrel{+}= s_r(j)\, x_j$$

Every coordinate lands in exactly one bucket per row, so the whole vector is folded into $d \cdot w$ numbers regardless of how large $D$ is.

**Recovery.** To estimate coordinate $j$, read the bucket it hashed into and undo its sign: $\hat{x}_j^{(r)} = s_r(j) \cdot \text{table}[r, h_r(j)]$. That bucket contains $x_j$ *plus* every other coordinate that collided with it, each carrying an independent random sign. Because those signs are symmetric, the contamination has **mean zero** — the estimate is unbiased — and the final estimate takes the **median across the $d$ rows**, so a coordinate needs to be unlucky in a majority of independent rows before its estimate is badly wrong.

Three properties matter for this project:

- **Linearity.** $\text{sketch}(x) + \text{sketch}(y) = \text{sketch}(x + y)$ when rows share hash functions. Sketches can be summed, averaged, and scaled *while still compressed*. This is the property that makes Count Sketch usable in federated learning at all: the server can aggregate ten clients' compressed messages directly and decompress only once, instead of decompressing ten times and averaging the errors.
- **Error scales with width, not dimension.** The noise in a recovered coordinate is governed by how much of the vector's total energy shares its buckets — roughly $\|x\|_2/\sqrt{w}$ — not by $D$. Doubling the width halves the variance; the ambient dimension barely enters. This is the sublinear-space promise.
- **It is a heavy-hitter structure.** Large-magnitude coordinates stand out over the collision noise and are recovered well. Small coordinates are, in effect, buried under the noise floor produced by everything else in their bucket. Count Sketch preserves *the shape of the extremes*, not the fine detail of the bulk.

That last property is the crux of the entire experiment. Count Sketch is excellent when a signal is dominated by a few big entries. A neural network update from a client with skewed local data is not obviously such a signal — and testing that is the point.

---

## 3. The experiment

### Goal

Determine, empirically, how the accuracy of federated training degrades as a function of the compression ratio applied to client updates, and identify *why* it degrades — whether the failure is a gentle, quantifiable loss of fidelity, or a structural breakdown of the estimator. The theory guarantees an unbiased estimate at any width. The experiment asks whether unbiasedness is worth anything once the estimate is fed back into a learning loop that runs for fifty rounds.

### Setup

**Task and data.** EMNIST *balanced*: 47 classes covering digits, uppercase letters, and eleven visually distinct lowercase letters, in 28×28 grayscale. EMNIST was chosen over MNIST deliberately — 47 classes is a genuinely harder problem than 10, so the model's update vectors carry more structure and there is more room for compression to do visible damage. (Torchvision stores EMNIST images transposed; the loader corrects this before normalization.)

**Heterogeneity.** Data is partitioned *pathologically non-IID*: training indices are sorted by label, cut into 400 contiguous shards, and each of the 100 clients receives 4 randomly chosen shards. A client therefore sees only a handful of the 47 classes. This is the standard hard case for federated learning, and it is the honest one — real devices do not hold uniform samples of the world. It also matters specifically for sketching: heterogeneous clients produce update vectors that *disagree*, so the per-client updates that get sketched are less similar to each other than IID updates would be.

**Model.** A compact CNN — two convolutional layers (32 and 64 channels, 5×5, each followed by 2×2 max-pooling) into two fully connected layers (1024→128→47) — totaling **189,359 trainable parameters**. Small enough to train 400 client-rounds in reasonable time, large enough that compressing its update vector is a meaningful operation rather than a rounding error.

**Federation protocol.** 100 clients, 10 sampled uniformly per round, 1 local epoch of SGD per selected client (lr 0.01, batch size 32), 50 communication rounds, equal weighting in the average. Test accuracy on the full held-out set is recorded after every round. Every run — baseline and all seven sketch configurations — uses identical seeds, so all runs draw the same client sample sequence and start from the same initialization. The *only* variable across runs is the compression.

**Sketch configuration.** Rows are fixed at $d = 10$ for every run, and the target compression ratio determines the width: $w = \lfloor D / (d \cdot R) \rceil$. Holding rows constant while varying width isolates the variable that theory says controls the error, and keeps the median-of-10 estimator identical across all runs so that differences in outcome are attributable to bucket pressure alone.

| Target ratio | Sketch width $w$ | Floats sent per client | Uplink per round (10 clients) |
|---|---|---|---|
| 1× (FedAvg) | — | 189,359 | 7.57 MB |
| 2× | 9,468 | 94,680 | 3.79 MB |
| 5× | 3,787 | 37,870 | 1.52 MB |
| 10× | 1,894 | 18,940 | 0.76 MB |
| 50× | 379 | 3,790 | 0.15 MB |
| 100× | 189 | 1,890 | 0.08 MB |
| 200× | 95 | 950 | 0.04 MB |
| 1000× | 19 | 190 | 0.008 MB |

At the extreme, 190 numbers are being asked to represent a 189,359-dimensional vector: roughly 10,000 parameters share every bucket.

**What is measured.** (a) Final and per-round test accuracy for every configuration; (b) exact payload size per round, so accuracy is comparable against bandwidth rather than against ratio alone; (c) a direct, training-independent comparison of a true gradient against its sketched reconstruction at each width, which separates *estimator quality* from *learning outcome*.

---

## 4. Design choices worth noting

**The whole update is one flat vector.** Each client's update is computed as `new_params − old_params` and all layers are concatenated into a single 189,359-dimensional tensor before sketching (`src/client.py`). This keeps the sketch layer-agnostic — one hash family covers the entire network — but it is a real modeling decision with consequences: convolutional filters and the fully connected layer share buckets, so the 131,072-parameter `fc1` block dominates the noise floor for the 832 parameters in `conv1`. A per-layer sketch would trade this off differently.

**Updates, not weights.** Clients transmit deltas because both parties already know the round's starting point. Deltas are also smaller in magnitude and more nearly centered than raw weights, which suits a zero-mean-noise estimator.

**Aggregation happens in sketch space.** The server averages the ten received $10 \times w$ matrices and calls `unsketch` exactly once (`src/server.py:sketch_aggregate`). This is the linearity property being used rather than merely cited: decompressing each client separately and then averaging would inject ten independent reconstruction errors instead of one, and would also defeat the point — the server never needs to see an individual client's update.

**Hash functions are derived from a shared seed, never transmitted.** `CountSketch` builds its bucket assignments and signs from `numpy.random.RandomState(seed + row)`, so client and server construct byte-identical hash functions independently. Only the table travels. Had the hash functions been part of the payload, the "compression" would have been fictional — the mapping alone is $d \cdot D$ integers, larger than the vector being compressed.

**Median, not mean, across rows.** Collision noise is heavy-tailed: most rows are clean and a few are badly contaminated. The median discards the contaminated rows; a mean would average them in. This is the difference between Count Sketch and Count-Min-style estimators, and it is why the estimator survives at moderate widths.

**Sketching is a scatter, not a loop.** The per-coordinate accumulation is expressed as `scatter_add_` over the precomputed bucket index tensor, and recovery as a gather — so both directions are single vectorized ops per row on whatever device the input tensor lives on (`src/sketch.py`). The class stores its tables on CPU and migrates them to match the input, so the same code runs on GPU or CPU with no branching.

**Every run is serialized, including per-round history.** Each configuration writes a `.pt` file holding the full accuracy trajectory, the achieved compression ratio, the payload accounting, and the final weights. Figures are therefore reproducible from saved artifacts without retraining (`regenerate_plots.py`), and the trained models themselves can be reloaded for qualitative inspection — which is what makes the prediction grid below possible.

---

## 5. Results

### 5.1 Accuracy versus compression

![Final test accuracy against compression ratio](figures/trial3/accuracy_vs_compression.png)

The baseline reaches 0.420 after 50 rounds. Against it:

| Ratio | Final | Peak | Retained |
|---|---|---|---|
| 2× | 0.388 | 0.404 | 92% |
| 5× | 0.352 | 0.376 | 84% |
| 10× | 0.328 | 0.328 | 78% |
| 50× | 0.213 | 0.234 | 51% |
| 100× | 0.175 | 0.209 | 42% |
| 200× | 0.133 | 0.140 | 32% |
| 1000× | 0.021 | 0.021 | 5% |

Three things are significant here.

**There is no free lunch, even at 2×.** Halving the payload costs 3.2 accuracy points. Classical sketching results are usually stated for a *single* estimation, where a small unbiased error is genuinely negligible. Here the estimate is applied to the model and the *next* round's updates are computed from the perturbed model. The error is not observed once; it is integrated fifty times. Sketching inside a feedback loop is a strictly harder regime than sketching a static stream.

**Degradation is linear in log-compression across two orders of magnitude.** Between 2× and 200× the curve is close to a straight line on the log axis — roughly 3.8 accuracy points lost per doubling of the compression ratio. This is the sublinear-space promise showing up as a measurable exchange rate rather than as a threshold: the algorithm does not have a "safe" operating point that it defends and then abandons, it trades continuously. Practically, 10× is the knee — 78% of the accuracy for 10% of the bandwidth is a defensible trade; 100× is not.

**1000× is not a bad model, it is a dead one.** 0.021 is 1/47 — exactly chance on a 47-class balanced problem. At 19 buckets per row, roughly 10,000 parameters collide in every bucket, the noise term overwhelms the signal in every row simultaneously, and the median has nothing clean to select. This point is the most informative on the plot: it marks where the estimator stops being a noisy version of the answer and becomes independent of it. The transition between 200× (still learning, badly) and 1000× (learning nothing) is where the sketch's width falls below the effective information content of the update.

### 5.2 Reconstruction quality, isolated from training

![True gradient coordinates against Count Sketch reconstructions](figures/trial3/gradient_recovery.png)

Each panel plots ~4,000 randomly sampled coordinates of one exact gradient against their Count Sketch reconstruction; the dashed line is perfect recovery. This measurement is deliberately *outside* the training loop — a single sketch/unsketch of a single true gradient — so it shows estimator quality with no compounding.

The panels explain the accuracy curve mechanically:

- **At 2×**, the cloud lies along the diagonal. The sketch is a genuine, if noisy, reconstruction.
- **By 10×**, the cloud has rotated toward vertical. The estimates are still *centered* on the truth — unbiasedness holds, exactly as theory promises — but the spread at a given true value now exceeds the range of true values themselves. The estimator is unbiased and nearly uninformative at the same time. This is the single most important thing the figure shows: **unbiasedness is a statement about the mean of the error, not about its usefulness.**
- **From 50× onward**, the vertical band is all that remains for the bulk of coordinates, and the y-axis range *grows* — from ±0.5 at 2× to ±4 at 1000× — while the true gradient range stays fixed at roughly ±0.5. Shrinking the sketch does not merely lose detail; it actively injects energy that was never in the signal, because each bucket now sums thousands of coordinates and the surviving imbalance is large.
- **The heavy hitters survive longest.** The isolated large-magnitude points at the right edge stay near the diagonal well past the ratio at which the bulk has collapsed. This is Count Sketch behaving exactly as designed — and it also diagnoses the failure. The algorithm is preserving the extremes of a vector whose learning signal is *not* concentrated in its extremes. Non-IID client updates after a full local epoch are dense: thousands of parameters move by similar small amounts. A heavy-hitter structure is close to the worst possible summary for a dense vector, which is why accuracy falls faster than a naive reading of the error bound would suggest.

### 5.3 What the compressed models actually do

![Sample predictions across compression ratios](figures/trial3/prediction_grid.png)

The same ten test characters, classified by every trained model. Green is correct, red is wrong.

This figure answers a question the accuracy curve cannot: *what kind of wrong* does a compressed model become?

- **FedAvg and CS@2/CS@5** make plausible, human-like errors — `C` read as `f`, `B` read as `A`, lowercase `q` read as `9`. These are confusions between genuinely similar glyphs. The model has learned real visual structure and is failing at the boundaries, which is what an under-trained but functioning classifier looks like.
- **CS@50 through CS@200** still produce *varied* predictions, but the errors stop tracking visual similarity — `E` read as `L`, `X` read as `f`. The decision boundaries are being displaced by accumulated sketch noise rather than being merely imprecise.
- **CS@1000 predicts `t` for every input.** The model has collapsed to a constant function. It has not learned a poor representation; it has learned nothing, and the classifier is just emitting whatever bias the noise happened to leave largest. This is the visual signature of the 1/47 accuracy in the first figure, and it is the clearest possible demonstration that below a critical width the sketch transmits no gradient information at all.

Note also that the *baseline* misclassifies 4 of these 10 examples. Fifty rounds on a pathologically non-IID 47-class problem is not enough to train a strong model, and this grid is deliberately not curated to hide that. Every claim above is about the *relative* behavior of compressed runs against a baseline trained under identical conditions.

---

## 6. What this says about randomized algorithms in practice

The experiment supports the theory and complicates it in equal measure.

**The core guarantee holds and is visible.** Error is controlled by sketch width, the estimator stays centered at every width tested, and the median-of-rows construction does its job — the degradation between 2× and 200× is smooth and predictable, which is precisely what a variance bound in $w$ predicts. A 189,359-dimensional vector really is summarized in 18,940 numbers with enough fidelity to train, and that is not a trivial thing.

**But "unbiased" is a much weaker promise than it sounds.** Section 5.2 shows an estimator that is simultaneously unbiased and useless at 10×. Any application that consumes a single sketched estimate can afford to reason about the mean; an application that consumes the estimate, acts on it, and generates its next input from the result must reason about the variance, because the variance is what compounds. Fifty rounds of unbiased noise is not equivalent to one round of unbiased noise.

**Structural fit matters more than the ratio.** Count Sketch is a heavy-hitter algorithm. It was applied here to dense, heterogeneous, small-magnitude update vectors — a signal whose informative content is spread across the bulk, not concentrated in the tail. The 5.2 panels show it faithfully preserving exactly the part of the signal that mattered least. The lesson generalizes beyond this repository: choosing a randomized summary is choosing *which structure to preserve*, and a compression ratio quoted without reference to the signal's structure is not a meaningful number.

**Cheap randomness bought something real.** The hash functions cost nothing to transmit, the sketch is two vectorized ops, and there is no optimization, no calibration, no data-dependent tuning. At 10×, an algorithm with no knowledge of neural networks whatsoever retained 78% of the accuracy for 10% of the bandwidth. That is the actual case for randomized methods — not that they are lossless, but that the loss is *predictable, tunable by one parameter, and purchased with almost no computation.*

## 7. Limitations and honest caveats

- The baseline itself reaches only 0.420; all conclusions are relative to it, not to a well-trained EMNIST model.
- Single seed per configuration. The per-round accuracy trajectories fluctuate with standard deviation ≈0.017 over the last ten rounds, so differences smaller than a few points between adjacent ratios should not be over-read. The order-of-magnitude trends are far larger than this noise.
- No **error feedback**. The most important known fix for compressed federated learning is to have each client accumulate its own compression residual and add it back the following round, which recovers much of the lost accuracy at high ratios (the FetchSGD line of work). This implementation compresses statelessly, so the numbers above should be read as a *floor* for what Count Sketch can do, not a ceiling.
- Rows are fixed at 10 throughout. The $d$/$w$ trade-off at a fixed budget is unexplored and would likely shift the knee.
- Payload accounting assumes dense float32 in both directions and ignores the download cost, which compression does not address.

## 8. Repository map

```text
fedlearn_countsketch/
├── src/
│   ├── sketch.py       # CountSketch: scatter-based sketch, gather + median unsketch
│   ├── client.py       # local SGD, delta computation, flattening
│   ├── server.py       # FedAvg mean, sketch-space aggregation, update application
│   ├── data.py         # EMNIST loading, transpose fix, non-IID shard partitioning
│   ├── model.py        # SmallCNN
│   ├── trainer.py      # the federated round loop, payload accounting, serialization
│   └── config.py       # every hyperparameter, in one place
├── plots/plotting.py   # the three figures
├── run_experiment.py   # full sweep: baseline + all ratios + figures
├── regenerate_plots.py # figures from saved .pt results, no retraining
├── results/            # per-run accuracy histories, payloads, final weights
├── figures/trial3/     # the figures presented above
└── notebooks/          # standalone Count Sketch demo on synthetic vectors:
                        # recovery behavior, linearity, error vs. sketch size
```

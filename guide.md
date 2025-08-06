# NeuCube‑Py Tutorial

This tutorial provides a comprehensive guide to NeuCube‑Py, a Python library for spiking‑reservoir computing on time‑series data. You’ll learn:

1. **Overview & Purpose**
2. **Typical Workflow**
3. **Module Breakdown & Parameters**
4. **Data Requirements**
5. **Step‑by‑Step Quick Start**
6. **Cross‑Validation & Hyperparameter Optimization**


## 1. Overview & Purpose

NeuCube‑Py enables researchers and practitioners to rapidly prototype spiking‑neuron reservoir models on sequential data (e.g., EEG, sensor streams, financial time‑series). Key advantages:

* **Biologically inspired**: Leverages spiking neurons and online learning (STDP).
* **Flexible**: Swap encoders, reservoir topologies, samplers, and classifiers.
* **Scalable**: GPU/MPS support for large reservoirs.

## 2. Typical Workflow

Below is the end-to-end conceptual pipeline in NeuCube-Py. Each step transforms the data, moving from raw signals to a static feature representation suitable for classification:

1. **Data Preparation**
   Gather and organize your observations as sequences. Each sample is a trial, time series, or any ordered data (e.g., flattened image frames).

   * Input: Raw signals of shape `[n_samples, n_time_steps, n_features]`.

2. **Spike Encoding**
   Convert continuous values into discrete events (spikes) that mimic how neurons communicate. By thresholding or detecting changes, we encode the dynamics of the signal as a temporal pattern of spikes.

   * Output: Binary (or ternary) spike trains of the same shape.

3. **Reservoir Simulation**
   Project the spike trains into a high-dimensional dynamical system. The recurrent network (reservoir) nonlinearly integrates past spikes, creating rich temporal representations that implicitly capture memory and temporal correlations.

   * Output: Spike activity tensor `[n_samples, n_time_steps, n_neurons]`.

4. **State-Vector Sampling**
   Summarize the reservoir’s evolving state into a fixed-length vector. This distills the reservoir’s dynamic response into features that machines can readily consume (e.g., total spike counts, firing rates, binned activity, or temporal statistics).

   * Output: Feature matrix `[n_samples, n_features_state]`.

5. **Readout / Classification**
   Use a simple linear or nonlinear classifier on the state vectors. Because the reservoir has already untangled complex temporal patterns into a high-dimensional space, even a basic readout can achieve strong performance.

Although illustrated for time-series, any data modality (images, audio, video) can be encoded as spike trains—making NeuCube-Py a versatile framework for spiking-reservoir computing.

## 3. Module Breakdown & Parameters

In this section, we break down each core module of NeuCube-Py, describe its purpose, key components, and list all configurable parameters.

### 3.1 Encoders

Transform continuous-valued time series into spike trains.

* **Purpose:** Convert analog signals into binary (or ternary) spikes suitable for spiking-neuron reservoirs.
* **Key Classes:**

  * `Delta` – Generates spikes when the difference between consecutive values exceeds a threshold.
  * `StepForward` – Emits +1/–1 spikes when the signal steps up/down by the threshold.

| Parameter   | Type  | Default | Description                                                        |
| ----------- | ----- | ------- | ------------------------------------------------------------------ |
| `threshold` | float | 0.1     | Minimum change required to generate a spike (absolute difference). |

### 3.2 Reservoir

A 3D spiking-neuron network with recurrent small-world connectivity.

* **Purpose:** Aggregate temporal patterns in a high-dimensional spiking state space.
* **Key Components:**

  * **Neuron Positions:** Automatically generated grid or user-supplied coordinates.
  * **Recurrent Weights:** Small-world adjacency matrix with excitatory/inhibitory connections.
  * **Input Weights:** Connections from encoder outputs into the reservoir.
  * **Online Learning:** Optional STDP rule during simulation.

| Parameter      | Type          | Default    | Description                                                                            |
| -------------- | ------------- | ---------- | -------------------------------------------------------------------------------------- |
| `cube_shape`   | tuple of ints | (10,10,10) | Dimensions of the 3D neuron lattice.                                                   |
| `inputs`       | int           | —          | Number of input channels (must match encoder output features).                         |
| `c`, `l`       | float         | 0.4, 0.169 | Small-world connectivity: `c`=max connection probability, `l`=connection length scale. |
| `c_in`, `l_in` | float         | 0.9, 1.2   | Input→reservoir connectivity parameters (analogous to `c` and `l`).                    |
| `use_mps`      | bool          | False      | Use Apple MPS backend if available.                                                    |

### 3.3 Topology

Functions for generating reservoir and input connectivity.

* **Purpose:** Create biologically plausible small-world graphs based on neuron distances.
* **Key Function:**

  * `small_world_connectivity(dist, c, l)` – Returns a sparse adjacency matrix.

| Parameter | Type         | Description                                                  |
| --------- | ------------ | ------------------------------------------------------------ |
| `dist`    | Tensor (N×N) | Pairwise Euclidean distances between neurons.                |
| `c`       | float        | Maximum connection probability (peak of Gaussian).           |
| `l`       | float        | Decay rate (spread) of connection probability with distance. |

### 3.4 Samplers

Summarize reservoir spike trains into fixed-length state vectors.

* **Purpose:** Produce features for downstream classifiers by reducing the time dimension.
* **Key Classes:**

  * `SpikeCount` – Counts spikes per neuron.
  * `MeanFiringRate` – Computes average firing rate.
  * `Binning` / `TemporalBinning` – Aggregates spikes into temporal bins.
  * `ISIstats` – Calculates statistics of inter-spike intervals.
  * `DeSNN` – Rank-based readout emphasizing early spikes.

| Class             | Parameters                               | Output Shape                                           |
| ----------------- | ---------------------------------------- | ------------------------------------------------------ |
| `SpikeCount`      | —                                        | (batch\_size, n\_neurons)                              |
| `MeanFiringRate`  | —                                        | (batch\_size, n\_neurons)                              |
| `Binning`         | `bin_size` (int)                         | (batch\_size, n\_neurons × (n\_time\_steps/bin\_size)) |
| `TemporalBinning` | `bin_size` (int)                         | (batch\_size, n\_bins × n\_neurons)                    |
| `ISIstats`        | —                                        | (batch\_size, n\_neurons)                              |
| `DeSNN`           | `alpha`, `mod`, `drift_up`, `drift_down` | (batch\_size, n\_neurons)                              |

### 3.5 Pipeline

High‑level API combining reservoir, sampler, and sklearn classifier.

* **Purpose:** Simplify end‑to‑end experiments with `.fit()` and `.predict()`.
* **Key Methods:**

  * `fit(X_train, y_train, train=False, learning_rule, verbose)` – Simulate reservoir, sample, then train classifier.
  * `predict(X_test)` – Simulate, sample, then return classifier predictions.

| Parameter         | Type                   | Description                                                  |
| ----------------- | ---------------------- | ------------------------------------------------------------ |
| `res_model`       | `Reservoir` instance   | Spiking reservoir object                                     |
| `sampling_method` | `Sampler` instance     | Feature extractor (e.g., `SpikeCount()`)                     |
| `classifier`      | scikit‑learn estimator | Any sklearn-compatible model (e.g., SVM, LogisticRegression) |

### 3.6 Utils & Visualization

Helper functions for inspection and plotting.

* **Purpose:** Quickly check reservoir structure and spike activity patterns.
* **Functions:**

  * `print_summary(info_ls)` – Tabular summary in console.
  * `spike_raster(spike_activity)` – 2D raster plot of spikes.
  * `plot_connections(reservoir)` – 3D plot of excitatory/inhibitory connections.

## 3. Data Requirements Data Requirements

NeuCube‑Py expects input data as a PyTorch tensor of shape:

```
X: [n_samples, n_time_steps, n_features]
```

* **n\_samples**: Number of sequences (e.g., trials).
* **n\_time\_steps**: Length of each sequence (time axis).
* **n\_features**: Number of parallel channels (e.g., EEG electrodes).

Targets (**y**) can be any 1D array of length `n_samples` for classification/regression.

## 4. Step‑by‑Step Quick Start

### 4.1 Installation

```bash
pip install git+https://github.com/KEDRI-AUT/NeuCube-Py.git
```

### 4.2 Import & Encode

```python
import torch
from neucube.encoder import Delta

# Suppose raw_data is a NumPy array or torch.Tensor of shape (N, T, F)
X_raw = torch.tensor(raw_data, dtype=torch.float)
enc = Delta(threshold=0.8)
X_spikes = enc.encode_dataset(X_raw)
```

### 4.3 Build Reservoir

```python
from neucube import Reservoir

res = Reservoir(
    cube_shape=(10,10,10),
    inputs=X_spikes.shape[2],
    c=0.4, l=0.17,
    c_in=0.9, l_in=1.2,
    use_mps=False
)
res.summary()
```

### 4.4 Simulate & Train

```python
from neucube.training import STDP

spikes = res.simulate(
    X_spikes,
    mem_thr=0.1,
    refractory_period=5,
    train=True,
    learning_rule=STDP(),
    verbose=True
)
```

### 4.5 Sample & Classify

```python
from neucube.sampler import SpikeCount
from neucube.validation import Pipeline
from sklearn.svm import SVC

sampler = SpikeCount()
pipe = Pipeline(res_model=res, sampling_method=sampler, classifier=SVC(kernel='linear'))
pipe.fit(X_spikes, y, train=True, learning_rule=STDP(), verbose=False)
y_pred = pipe.predict(X_spikes)
```

---

## 5. Cross‑Validation & Hyperparameter Optimization

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from neucube.encoder import Delta
from neucube import Reservoir
from neucube.sampler import SpikeCount
from neucube.validation import Pipeline
from neucube.training import STDP

def evaluate_cv(X, y, param_grid, cv_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    best = {'score':0}

    for c in param_grid['c']:
      for l in param_grid['l']:
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Encode & reservoir
            enc = Delta(threshold=param_grid.get('threshold', 0.8))
            X_tr_sp = enc.encode_dataset(X_tr)
            X_te_sp = enc.encode_dataset(X_te)

            res = Reservoir(inputs=X_tr_sp.shape[2], c=c, l=l)
            pipe = Pipeline(res_model=res, sampling_method=SpikeCount(), classifier=SVC(kernel='linear'))
            pipe.fit(X_tr_sp, y_tr, train=True, learning_rule=STDP(), verbose=False)
            y_pred = pipe.predict(X_te_sp)
            scores.append((y_pred == y_te).mean())

        avg_score = np.mean(scores)
        if avg_score > best['score']:
            best = {'c':c, 'l':l, 'score':avg_score}

    return best

# Example grid
grid = {'c':[0.2,0.3], 'l':[0.1,0.17]}
best_params = evaluate_cv(X_raw, y, grid)
print(f"Best params: {best_params}")
```

This routine:

1. Encodes with different thresholds
2. Builds reservoirs with varied `c`/`l` values
3. Trains & tests via StratifedKFold
4. Returns optimal combo

---

**Happy spiking!** Feel free to customize encoders, connectivity, samplers, and classifiers to suit your domain-specific data and tasks.

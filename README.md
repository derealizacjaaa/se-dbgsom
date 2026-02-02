# DBGSOM - Directed Batch Growing Self-Organizing Map

A neural network library for unsupervised learning, clustering, and dimensionality reduction. Extends Kohonen SOMs with intelligent growth mechanisms and directed error distribution.

**Key Innovation**: The network dynamically grows by directing errors from internal neurons to boundary neurons - no need to specify grid dimensions upfront.

## Tech Stack

- **Julia 1.9+** (core): BLAS-optimized SOM algorithm
- **Python 3.8+** (wrapper): High-level API via JuliaCall, visualization, clustering

## Installation

### Julia

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Python

```bash
pip install -e python/
pip install -e "python/[viz]"  # with matplotlib/seaborn
```

## Quick Start (Python)

```python
from dbgsom import SEDBGSOM, cluster_acute, cluster_leiden, cluster_vesanto

# Train
som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
som.fit(X)

# Cluster (pick one method)
labels = cluster_acute(som, X)                 # Non-convex shapes (moons, circles)
labels = cluster_leiden(som, X, n_clusters=3)  # Convex blobs
labels = cluster_vesanto(som, X)               # Convex, auto-k via Davies-Bouldin

# Get sample-level labels
from dbgsom import assign_cluster_labels
sample_labels = assign_cluster_labels(som, X, labels)
```

## Quick Start (Julia)

```julia
using BasicDBGSOM

# StatisticalError method (recommended, data-adaptive)
som = DBGSOM(data; growth_threshold_method=StatisticalError(λ=1.5))
fit!(som, data; epochs=200)

# Results
labels = predict(som, data)        # BMU indices
coords = transform(som, data)      # 2D coordinates
```

> **Note**: Clustering is implemented in Python only. Julia clustering functions are stubs.

## Growth Threshold Methods

| Method | Formula | When to use |
|--------|---------|-------------|
| StatisticalError | `GT = λ * sqrt(Σ std_i²)` | Default - adapts to data |
| SpreadingFactor | `GT = -ln(sf) * d` | Fine-tuned control |

## Clustering Methods

| Method | Best for | Notes |
|--------|----------|-------|
| ACUTE | Non-convex (moons, circles) | Voronoi region analysis |
| Leiden | Convex blobs | Community detection |
| Vesanto | Convex, unknown k | Auto-selects k via Davies-Bouldin |

## Project Structure

```
src/                        # Julia core (BasicDBGSOM module)
├── core/                   # types.jl, topology.jl
├── optimization/           # distance.jl, neighborhood.jl, bmu.jl, batch_update.jl
├── algorithms/             # dbgsom.jl, growth.jl
├── visualization/          # plotting.jl
└── preprocessing/          # panel.jl (time-series/panel data)

python/dbgsom/              # Python wrapper
├── sedbgsom.py            # SEDBGSOM class (recommended)
├── wrapper.py             # DBGSOMWrapper (legacy)
├── clustering/            # acute.py, leiden.py, vesanto.py, spectral.py
└── visualization.py       # Matplotlib plotting

examples/                   # Datasets and demos
├── data/                  # iris, blobs, circles, two_moons, breast_cancer, etc.
├── python/                # Python demo scripts
└── output/                # Pre-generated output plots

world_panel/               # World economic panel data module
```

## Examples

See `examples/python/` for complete demos on 8 datasets. Each demo trains a DBGSOM, applies clustering, and generates visualizations.

## Two-Phase Training

1. **Coarse (50%)**: Large neighborhood, neuron growth enabled
2. **Fine (50%)**: Small neighborhood, weight refinement only

## Dependencies

**Julia** (Project.toml): LinearAlgebra, Statistics, Random, Graphs, CSV, DataFrames, MLJ, LoopVectorization, Plots

**Python** (pyproject.toml): numpy, pandas, juliacall; optional: matplotlib, seaborn, igraph, leidenalg

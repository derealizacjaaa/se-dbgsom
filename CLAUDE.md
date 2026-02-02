# DBGSOM - Directed Batch Growing Self-Organizing Map

## Project Overview

Neural network library for unsupervised learning, clustering, and dimensionality reduction. Extends Kohonen SOMs with intelligent growth mechanisms and directed error distribution.

**Key Innovation**: Network dynamically grows by directing errors from internal neurons to boundary neurons - no need to specify grid dimensions upfront.

## Tech Stack

- **Julia 1.9+** (core): BLAS-optimized SOM algorithm
- **Python 3.8+** (wrapper): High-level API via JuliaCall, visualization, clustering

## Directory Structure

```
src/                        # Julia core (BasicDBGSOM module)
├── core/                   # types.jl, topology.jl
├── optimization/           # distance.jl, neighborhood.jl, bmu.jl, batch_update.jl, nan_handling.jl
├── algorithms/             # dbgsom.jl, growth.jl
├── visualization/          # plotting.jl
└── preprocessing/          # panel.jl (time-series/panel data)

test/                       # Julia test suite
├── runtests.jl            # Test runner
├── test_nan_handling.jl   # NaN-aware training tests
└── test_qe_te.jl          # QE/TE quality metric tests

python/dbgsom/              # Python wrapper
├── sedbgsom.py            # SEDBGSOM class (recommended)
├── wrapper.py             # DBGSOMWrapper (legacy)
├── clustering/            # acute.py, leiden.py, vesanto.py, spectral.py - REAL clustering
└── visualization.py       # Matplotlib plotting

examples/
├── data/                  # iris, blobs, circles, two_moons, breast_cancer, parkinsons, wine, glass, ecoli
├── python/                # Python demos
└── output/                # Pre-generated output plots

world_panel/               # World economic panel data demo
```

## Gotchas

### Julia clustering is stubbed out

`src/clustering/acute.jl` and `leiden.jl` are **mock files** that return empty results. All real clustering functionality is in **Python only**:

```python
# CORRECT: Use Python clustering
from dbgsom import cluster_acute, cluster_leiden, cluster_vesanto
labels = cluster_acute(som, X)
```

### Windows/OneDrive path issues

JuliaCall can fail with OneDrive paths containing spaces. If you see `LoadError: SystemError: opening file`, try moving the project to a path without spaces.

## Quick Start

```bash
# Julia setup
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Python setup
pip install -e python/
pip install -e "python/[viz]"  # with matplotlib/seaborn
```

## Python API (Recommended)

```python
from dbgsom import SEDBGSOM, cluster_acute, cluster_leiden, cluster_vesanto

# Train
som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
som.fit(X)

# Cluster (pick one method)
labels = cluster_acute(som, X)              # Non-convex shapes (moons, circles)
labels = cluster_leiden(som, X, n_clusters=3)  # Convex blobs
labels = cluster_vesanto(som, X)            # Convex, auto-k via Davies-Bouldin

# Get sample-level labels
from dbgsom import assign_cluster_labels
sample_labels = assign_cluster_labels(som, X, labels)
```

## Julia API

```julia
using BasicDBGSOM

# StatisticalError method (recommended, data-adaptive)
som = DBGSOM(data; growth_threshold_method=StatisticalError(λ=1.5))
fit!(som, data; epochs=200)

# Results
labels = predict(som, data)        # BMU indices
coords = transform(som, data)      # 2D coordinates

# Note: Use Python for clustering - Julia clustering is stubbed
```

## Key Concepts

### Growth Threshold Methods
| Method | Formula | When to use |
|--------|---------|-------------|
| StatisticalError | `GT = λ × sqrt(Σ std_i²)` | Default - adapts to data |
| SpreadingFactor | `GT = -ln(sf) × d` | Fine-tuned control |

### Clustering Methods
| Method | Best for | Notes |
|--------|----------|-------|
| ACUTE | Non-convex (moons, circles) | Voronoi region analysis |
| Leiden | Convex blobs | Community detection |
| Vesanto | Convex, unknown k | Auto-selects k via Davies-Bouldin |

### Two-Phase Training
1. **Coarse (50%)**: Large neighborhood, neuron growth enabled
2. **Fine (50%)**: Small neighborhood, weight refinement only

### NaN-Aware Training
When input data contains NaN values, the library automatically:
1. Computes per-variable missingness rates (`MissingnessInfo`)
2. Uses missingness-weighted distances for BMU finding (high-missingness dims contribute less)
3. Imputes poorly-observed neuron dimensions from SOM grid neighbors
4. Stores `MissingnessInfo` on the trained model for consistent prediction

All code paths (training, pruning, error accumulation, prediction) use the same weighted metric via `dispatch_find_bmus`.

## Important Files

| File | Purpose |
|------|---------|
| `src/DBGSOM.jl` | Julia module entry point |
| `src/core/types.jl` | Neuron, SOMTopology, DBGSOM, MissingnessInfo structs |
| `src/algorithms/dbgsom.jl` | fit!, predict, transform |
| `src/optimization/bmu.jl` | BMU finding, dispatch_find_bmus |
| `src/optimization/nan_handling.jl` | NaN-weighted distances, neighbor imputation |
| `python/dbgsom/sedbgsom.py` | SEDBGSOM class |
| `python/dbgsom/clustering/` | **Real clustering implementations** |

## Code Conventions

### Julia
- `T<:AbstractFloat` for numeric generics
- Functions modifying state end with `!` (fit!, grow!)
- BLAS for distance computations (falls back to pure Julia loops for NaN-weighted distances)
- Neurons in `Dict{Tuple{Int,Int}, Neuron}`
- Use `dispatch_find_bmus(topo, X, miss_info)` for NaN-aware BMU finding (centralizes dispatch logic)

### Running Tests
```bash
julia --project=. test/runtests.jl
```

### Python
- NumPy arrays: samples x features
- JuliaCall handles type conversion
- Clustering returns neuron labels; use `assign_cluster_labels()` for samples

## Dependencies

**Julia** (Project.toml): LinearAlgebra, Statistics, Random, Graphs, CSV, DataFrames, MLJ, LoopVectorization, Plots

**Python** (pyproject.toml): numpy, pandas, juliacall; optional: matplotlib, seaborn, igraph, leidenalg

# DBGSOM

Directed Batch Growing Self-Organizing Map with Statistical Error growth threshold.

The network dynamically grows neurons on a hexagonal grid by directing quantization errors from internal neurons to boundary neurons.

## Implementations

| Language | Location | Dependencies |
|----------|----------|--------------|
| Julia | `src/` | Core implementation |
| Python | `python/` | Wraps Julia via juliacall |

## Installation

### Julia

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Python

```bash
pip install -e python/
pip install -e "python/[viz]"  # matplotlib/seaborn
```

## Usage

### Julia

```julia
using DBGSOM

som = DBGSOM.BasicDBGSOM(data; lambda=1.5, max_neurons=100)
DBGSOM.fit!(som, data; epochs=200)

labels = DBGSOM.predict(som, data)
coords = DBGSOM.transform(som, data)
```

### Python

```python
from dbgsom import SEDBGSOM, cluster_vesanto, assign_cluster_labels

som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
som.fit(X)

neuron_labels = cluster_vesanto(som, X, n_clusters=3)
sample_labels = assign_cluster_labels(som, X, neuron_labels)
# or auto-detect k:
neuron_labels = cluster_vesanto(som, X)
```


## Algorithm

Two-phase batch training:
1. **Coarse (50%)**: Large neighborhood, neuron growth enabled
2. **Fine (50%)**: Small neighborhood, weight refinement only

**Growth threshold**: `GT = lambda * sqrt(sum(std_i^2))`

- Lower lambda = more neurons
- Higher lambda = fewer neurons

## Vesanto Clustering

Two-level clustering of SOM prototypes:
1. Train SOM to produce prototype vectors
2. Cluster prototypes using Ward's hierarchical or K-means
3. Auto-select k via Davies-Bouldin (minimize) or silhouette (maximize)

## Project Structure

```
src/                    Julia core
├── core/               types.jl, topology.jl
├── optimization/       distance.jl, neighborhood.jl, bmu.jl, batch_update.jl
├── algorithms/         dbgsom.jl, growth.jl
├── visualization/      plotting.jl
└── preprocessing/      panel.jl

python/dbgsom/          Python wrapper
├── sedbgsom.py         Main class
├── wrapper.py          Julia bridge (legacy)
├── clustering/         vesanto.py, api.py, metrics.py
└── visualization.py    Plotting

examples/
├── data/               iris, wine, glass, ecoli, breast_cancer, circles, wholesale
└── python/             Demo scripts
```

## Evaluation Metrics

- Silhouette score
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- Purity

## Dependencies

**Julia**: LinearAlgebra, Statistics, Graphs, DataFrames, MLJ, LoopVectorization, Plots

**Python**: numpy, pandas, juliacall; optional: matplotlib, seaborn

## License

MIT

# DBGSOM Runbook

Deployment, operations, and troubleshooting guide.

## Deployment

### Installing as a Python Package

```bash
# From PyPI (when published)
pip install dbgsom

# From source
pip install git+https://github.com/maxuser/dbgsom.git#subdirectory=python

# Local development
pip install -e "python/[viz]"
```

### Julia Environment Setup

The Python wrapper automatically sets up Julia on first use. To pre-configure:

```python
from dbgsom import setup_julia_environment, is_julia_ready, get_julia_version

# Check status
print(f"Julia ready: {is_julia_ready()}")
print(f"Julia version: {get_julia_version()}")

# Force setup (downloads packages if needed)
setup_julia_environment()
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JULIA_PROJECT` | Julia project path | Auto-detected |
| `PYTHON_JULIAPKG_OFFLINE` | Disable Julia package downloads | `false` |

## Common Operations

### Training a Model

```python
from dbgsom import SEDBGSOM

som = SEDBGSOM(
    lambda_=1.0,      # Growth threshold (0.5=more neurons, 2.0=fewer)
    max_neurons=100,  # Upper limit on neuron count
    n_iter=200,       # Training iterations
    random_state=42   # Reproducibility
)
som.fit(X)

# Check model info
print(som.summary())
print(f"Neurons: {som.n_neurons_}")
print(f"Quantization error: {som.quantization_error(X):.4f}")
print(f"Topographic error: {som.topographic_error(X):.4f}")
```

### Clustering

```python
from dbgsom import cluster_vesanto, assign_cluster_labels

# Vesanto-Alhoniemi clustering with dendrogram
neuron_labels = cluster_vesanto(som, X)

# With specific number of clusters
neuron_labels = cluster_vesanto(som, X, n_clusters=3)

# Map to samples
sample_labels = assign_cluster_labels(som, X, neuron_labels)
```

### Visualization

```python
from dbgsom.visualization import DBGSOMVisualizer

viz = DBGSOMVisualizer('output/', 'my_dataset')
viz.save_all(som, X, labels=y, label_names=['Class A', 'Class B'])

# Individual plots
viz.plot_umatrix(som)
viz.plot_hit_counts(som, X)
viz.plot_component_planes(som, feature_names)
viz.plot_expanded_umatrix(som)
```

### Batch Processing

```python
from pathlib import Path
import pandas as pd

datasets = ['iris', 'wine', 'glass']
results = []

for name in datasets:
    df = pd.read_csv(f'data/{name}.csv')
    X = df.drop('target', axis=1)
    y = df['target'].values

    som = SEDBGSOM(lambda_=1.0, max_neurons=100, n_iter=200)
    som.fit(X)

    labels = cluster_vesanto(som, X)
    sample_labels = assign_cluster_labels(som, X, labels)

    results.append({
        'dataset': name,
        'neurons': som.n_neurons_,
        'qe': som.quantization_error(X),
        'te': som.topographic_error(X)
    })

print(pd.DataFrame(results))
```

## Monitoring

### Model Quality Metrics

| Metric | Good Range | Description |
|--------|------------|-------------|
| Quantization Error | Lower is better | Mean distance to BMU |
| Topographic Error | < 0.1 | Proportion with non-adjacent BMU1/BMU2 |
| Silhouette Score | > 0.5 | Cluster separation quality |
| Davies-Bouldin | < 1.0 | Cluster compactness (lower=better) |

### Memory Usage

```python
import sys

# Estimate memory
n_neurons = som.n_neurons_
n_features = som.n_features_in_
weights_mb = (n_neurons * n_features * 8) / (1024 * 1024)  # Float64
print(f"Approximate weight matrix: {weights_mb:.2f} MB")
```

### Performance Timing

```python
import time

start = time.time()
som.fit(X)
train_time = time.time() - start

start = time.time()
labels = cluster_vesanto(som, X)
cluster_time = time.time() - start

print(f"Training: {train_time:.2f}s")
print(f"Clustering: {cluster_time:.2f}s")
```

## Common Issues and Fixes

### Issue: Julia Setup Fails

**Symptoms**: `JuliaError` or `ModuleNotFoundError: No module named 'juliacall'`

**Fix**:
```bash
# Reinstall juliacall
pip uninstall juliacall
pip install juliacall>=0.9.14

# Clear Julia cache
rm -rf ~/.julia/compiled
rm -rf .venv/julia_env

# Restart Python and retry
python -c "from dbgsom import setup_julia_environment; setup_julia_environment()"
```

### Issue: Slow First Import

**Cause**: Julia JIT compilation on first use

**Fix**: Pre-compile Julia packages:
```bash
julia --project=. -e 'using DBGSOM'
```

### Issue: Out of Memory

**Symptoms**: `MemoryError` during training

**Fix**:
```python
# Reduce max_neurons
som = SEDBGSOM(lambda_=2.0, max_neurons=50, n_iter=100)

# Or subsample data
from sklearn.utils import resample
X_sample = resample(X, n_samples=5000, random_state=42)
som.fit(X_sample)
```

### Issue: Poor Clustering Results

**Symptoms**: Low silhouette, clusters don't match ground truth

**Diagnostic Steps**:
1. Check data preprocessing
2. Tune lambda_ parameter
3. Try different k values

```python
# 1. Check for NaN/Inf
print(f"NaN values: {np.isnan(X).sum()}")
print(f"Inf values: {np.isinf(X).sum()}")

# 2. Try different lambda values
for lam in [0.5, 1.0, 1.5, 2.0]:
    som = SEDBGSOM(lambda_=lam, max_neurons=100, n_iter=200)
    som.fit(X)
    print(f"lambda={lam}: {som.n_neurons_} neurons")

# 3. Evaluate clustering
from dbgsom import compute_all_metrics, cluster_vesanto

labels = cluster_vesanto(som, X)
sample_labels = assign_cluster_labels(som, X, labels)
bmus = som.predict(X)
metrics = compute_all_metrics(som, labels, X, y_true, bmus)
print(f"NMI={metrics['nmi']:.3f}, ARI={metrics['ari']:.3f}")
```

### Issue: Visualization Missing

**Symptoms**: `ImportError: matplotlib is required for visualization`

**Fix**:
```bash
pip install -e "python/[viz]"
# or
pip install matplotlib seaborn
```

### Issue: Windows Path Errors

**Symptoms**: Path-related errors on Windows

**Fix**: Use raw strings or forward slashes:
```python
# Good
viz = DBGSOMVisualizer(r'C:\output', 'dataset')
viz = DBGSOMVisualizer('C:/output', 'dataset')

# Bad
viz = DBGSOMVisualizer('C:\output', 'dataset')  # \o is escape sequence
```

## Rollback Procedures

### Downgrade Python Package

```bash
# Reinstall specific version
pip install dbgsom==0.1.0
```

### Reset Julia Environment

```bash
# Remove compiled artifacts
rm -rf ~/.julia/compiled/v1.*/DBGSOM

# Reinstantiate packages
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Clear All Caches

```bash
# Python
pip cache purge

# Julia
julia -e 'using Pkg; Pkg.gc()'

# Virtual environment (nuclear option)
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e "python/[dev,viz]"
```

## Performance Tuning

### Optimal Lambda Selection

| Data Characteristics | Recommended Lambda |
|---------------------|-------------------|
| High variance, many clusters | 0.5 - 0.8 |
| Balanced clusters | 0.8 - 1.2 |
| Low variance, few clusters | 1.5 - 2.0 |

### Iteration Guidelines

| Dataset Size | Recommended n_iter |
|-------------|-------------------|
| < 1000 samples | 100-200 |
| 1000-10000 | 200-500 |
| > 10000 | 500-1000 |

### Max Neurons Guidelines

| Dataset Size | Recommended max_neurons |
|-------------|------------------------|
| < 500 samples | 50-100 |
| 500-5000 | 100-300 |
| > 5000 | 300-1000 |

## Contact

For issues not covered here:
1. Check GitHub Issues
2. Review example scripts in `examples/`
3. Open a new issue with reproduction steps

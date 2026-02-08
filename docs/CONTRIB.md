# Contributing to DBGSOM

Development workflow, setup, and testing procedures.

## Project Overview

DBGSOM (Directed Batch Growing Self-Organizing Map) is a multi-language project:

| Language | Location | Purpose |
|----------|----------|---------|
| Julia | `src/` | Core algorithm implementation |
| Python | `python/dbgsom/` | Python wrapper via juliacall |

## Environment Setup

### Prerequisites

- **Julia**: 1.9+ (1.10 or 1.11 recommended)
- **Python**: 3.8+

### Julia Setup

```bash
# Install Julia dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Verify installation
julia --project=. -e 'using DBGSOM; println("Julia setup complete")'
```

### Python Setup

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install core package
pip install -e python/

# Install with visualization support
pip install -e "python/[viz]"

# Install with development tools
pip install -e "python/[dev]"

# Install all optional dependencies
pip install -e "python/[dev,viz]"
```

## Dependencies

### Julia (Project.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| LinearAlgebra | stdlib | Matrix operations |
| Statistics | stdlib | Statistical functions |
| Graphs | 1.9+ | Graph topology |
| DataFrames | 1.x | Data handling |
| LoopVectorization | 0.12+ | SIMD optimization |
| MLJ | 0.22+ | ML integration |
| Plots | 1.x | Visualization |
| CSV | 0.10+ | File I/O |
| JSON | 0.21.4+ | JSON serialization |
| StaticArrays | 1.6+ | Fixed-size arrays |
| StateSpaceModels | 0.7.0+ | Time series models |

### Python (pyproject.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Numerical arrays |
| pandas | >=1.3.0 | DataFrames |
| juliacall | >=0.9.14 | Julia bridge |
| matplotlib | >=3.5.0 | Visualization (optional) |
| seaborn | >=0.12.0 | Plot styling (optional) |
| pytest | >=7.0.0 | Testing (dev) |
| pytest-cov | >=4.0.0 | Coverage (dev) |

## Development Workflow

### 1. Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes to Julia core (src/)
# Make changes to Python wrapper (python/dbgsom/)

# Run examples to verify
python examples/python/iris_combined_demo.py
```

### 2. Code Organization

```
src/                        # Julia core
├── DBGSOM.jl              # Main module entry
├── core/                  # Types, topology
├── optimization/          # Distance, BMU, batch update
├── algorithms/            # DBGSOM growth logic
├── visualization/         # Julia plotting
└── preprocessing/         # Panel data handling

python/dbgsom/             # Python wrapper
├── __init__.py            # Public API exports (v0.2.0)
├── sedbgsom.py            # SEDBGSOM class (recommended)
├── wrapper.py             # Legacy DBGSOMWrapper
├── clustering/            # Clustering algorithms
│   ├── __init__.py        # Clustering exports
│   ├── api.py             # High-level functions
│   ├── vesanto.py         # Vesanto-Alhoniemi method
│   └── metrics.py         # Evaluation metrics
├── visualization.py       # Plot generation
├── preprocessor.py        # Data preprocessing
├── julia_setup.py         # Julia environment setup
└── utils.py               # Julia-Python data conversion

test/                      # Julia tests
├── runtests.jl            # Test runner
├── test_nan_handling.jl   # NaN handling tests
└── test_qe_te.jl          # QE/TE metric tests

examples/                  # Demo scripts
├── data/                  # Sample datasets
│   ├── iris.csv
│   ├── wine.csv
│   ├── glass.csv
│   ├── ecoli.csv
│   ├── breast_cancer.csv
│   ├── circles.csv
│   └── wholesale.csv
├── output/                # Generated plots
└── python/                # Python demos
    ├── iris_combined_demo.py
    ├── wine_combined_demo.py
    ├── circles_combined_demo.py
    ├── ecoli_combined_demo.py
    ├── breast_cancer_combined_demo.py
    ├── wholesale_combined_demo.py
    ├── world_panel_vesanto_demo.py
    └── plot_circles_data.py
```

### 3. API Guidelines

#### Recommended API (New Code)

```python
from dbgsom import SEDBGSOM, cluster_vesanto, assign_cluster_labels

# Training
som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
som.fit(X)

# Clustering (separate step)
neuron_labels = cluster_vesanto(som, X)   # With dendrogram
sample_labels = assign_cluster_labels(som, X, neuron_labels)
```

#### Legacy API (Still Supported)

```python
from dbgsom import DBGSOMWrapper

som = DBGSOMWrapper(sf=0.6)
som.fit(df)
clusters = som.cluster(df)
```

## Testing

### Running Tests

```bash
# Julia tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Python examples (no formal test suite yet)
python examples/python/iris_combined_demo.py
python examples/python/wine_combined_demo.py
```

### Running Examples

```bash
# All Python demos
python examples/python/iris_combined_demo.py
python examples/python/wine_combined_demo.py
python examples/python/circles_combined_demo.py
python examples/python/ecoli_combined_demo.py
python examples/python/breast_cancer_combined_demo.py
python examples/python/wholesale_combined_demo.py
python examples/python/world_panel_vesanto_demo.py
```

## Code Style

### Python

- Follow PEP 8
- Use type hints for function signatures
- Docstrings in NumPy format
- Maximum line length: 88 characters (Black formatter)

### Julia

- Follow Julia style guide
- Use docstrings with `@doc` or triple quotes
- Explicit type annotations for performance-critical code

## Commit Guidelines

```
<type>: <description>

Types: feat, fix, refactor, docs, test, chore, perf
```

Examples:
- `feat: add spectral clustering option`
- `fix: handle NaN values in U-matrix`
- `docs: update clustering API examples`

## Release Process

1. Update version in `python/dbgsom/__init__.py`
2. Update version in `python/pyproject.toml`
3. Update version in `Project.toml` (Julia)
4. Create git tag: `git tag v0.2.0`
5. Push tag: `git push origin v0.2.0`

## Getting Help

- Open an issue for bugs or feature requests
- Check existing examples in `examples/` directory
- Review docstrings in source code

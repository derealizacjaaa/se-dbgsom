"""
    BasicDBGSOM

A minimal, modular implementation of Directed Batch Growing Self-Organizing Map.

# Growth Threshold Methods
Two methods are available for controlling network growth:

1. **SpreadingFactor** (original): GT = -log(sf) × d
   - User-specified parameter sf in (0, 1]
   - sf ≈ 0.1: Conservative, sf ≈ 0.9: Aggressive

2. **StatisticalError** (data-adaptive): GT = λ × sqrt(Σ std_i²)
   - Adapts to dataset statistics automatically
   - λ ≈ 0.5: Aggressive, λ ≈ 2.0: Conservative

# Quick Start
```julia
using DBGSOM

# Method 1: Using spreading factor (original approach)
som = DBGSOM{Float64}(
    input_dim=10,      # required: dimension of input vectors
    sf=0.5,            # spreading factor (0.1=conservative, 0.9=aggressive)
    max_neurons=100,   # maximum neurons allowed
    n_iter=100         # training iterations
)

# Method 2: Using statistical error (data-adaptive)
som = DBGSOM{Float64}(
    input_dim=10,
    gt_method=StatisticalError(1.0),  # λ=1.0
    max_neurons=100,
    n_iter=100
)

# Train
fit!(som, X)  # X is input_dim × n_samples

# Predict
bmus = predict(som, X)      # BMU positions
coords = transform(som, X)  # 2 × n_samples coordinates
```

# Module Structure
- `core/types.jl`: Neuron, SOMTopology, DBGSOM structs, GrowthThresholdMethod types
- `core/topology.jl`: Grid navigation and topology queries
- `optimization/distance.jl`: Euclidean distance calculations
- `optimization/neighborhood.jl`: Gaussian kernel and sigma decay
- `optimization/bmu.jl`: Best Matching Unit finding
- `optimization/batch_update.jl`: Batch weight updates, growth threshold computation
- `algorithms/growth.jl`: Growth and pruning logic
- `algorithms/dbgsom.jl`: Main training algorithm

Reference: Vasighi & Amini (2017), "A directed batch growing approach to enhance
the topology preservation of self-organizing map"
"""
module DBGSOM

using LinearAlgebra
using Statistics
using Random
using Printf

# ============================================================================
# Exports
# ============================================================================

# Core types
export Neuron, SOMTopology, DBGSOM

# Growth threshold methods
export GrowthThresholdMethod, SpreadingFactor, StatisticalError

# Topology functions
export get_neighbors, grid_distance, euclidean_grid_distance
export is_boundary, get_empty_neighbors, get_occupied_neighbors
export grid_bounds, grid_extent
export add_neuron!, remove_neuron!
export get_weights, get_positions_and_weights
export n_neurons, input_dim, positions, get_neuron, has_neuron

# Distance functions
export euclidean, euclidean_squared
export euclidean_nan, euclidean_squared_nan
export pairwise_distances_squared, pairwise_distances_squared!
export pairwise_distances_squared_nan, pairwise_distances_squared_auto
export has_nan

# Neighborhood functions
export gaussian_neighborhood, compute_neighborhood_matrix
export compute_sigma, decay_sigma, compute_sigma_range

# BMU functions
export find_bmu, find_bmus, find_bmus_with_distances, dispatch_find_bmus

# Update functions
export update_weights_batch!
export compute_quantization_error, compute_growth_threshold

# NaN handling
export MissingnessInfo, compute_missingness_info
export euclidean_squared_nan_weighted, pairwise_distances_squared_nan_weighted
export impute_neuron_from_neighbors!, update_weights_batch_nan_impute!

# Growth functions
export reset_errors!, accumulate_errors!, distribute_errors!
export find_growth_candidates, insert_neuron!, grow!
export find_dead_neurons, prune_dead_neurons!

# Main API
export fit!, predict, transform, predict_with_distances

# Visualization
export PlottingData
export compute_umatrix, compute_ustar_matrix, compute_expanded_umatrix
export compute_component_plane, compute_all_component_planes
export compute_hit_counts, compute_topology_edges, get_plotting_data
export to_matrix, to_coordinate_arrays, to_edge_segments
export format_grid_ascii, format_umatrix_ascii, format_summary
export cluster_neurons_umatrix

# Quality metrics
export find_two_bmus, geodesic_distance
export compute_topographic_error

# Panel data preprocessing
export PanelData, SlidingWindowResult
export sliding_windows, n_windows, window_dim
export get_entity_windows, get_entity_trajectory, unique_entities, entity_counts
export get_window_feature_names
export panel_from_columns, panel_from_matrix

# ============================================================================
# Includes
# ============================================================================

# Core
include("core/types.jl")
include("core/topology.jl")

# Optimization
include("optimization/distance.jl")
include("optimization/neighborhood.jl")
include("optimization/bmu.jl")
include("optimization/batch_update.jl")
include("optimization/nan_handling.jl")

# Algorithms
include("algorithms/growth.jl")
include("algorithms/dbgsom.jl")

# Visualization
include("visualization/plotting.jl")

# Preprocessing
include("preprocessing/panel.jl")

end # module

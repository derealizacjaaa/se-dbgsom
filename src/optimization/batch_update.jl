"""
Batch weight update for BasicDBGSOM.
"""

# ============================================================================
# Batch Update
# ============================================================================

"""
    update_weights_batch!(topo, X, bmus, σ)

Batch weight update using Gaussian neighborhood.
Automatically handles NaN values by using per-dimension accumulators.

For each neuron j, the new weight is:
    w_j = Σᵢ h(j, bmu_i) * xᵢ / Σᵢ h(j, bmu_i)

where h(j, k) = exp(-d²_jk / (2σ²)) is the neighborhood function.

# Arguments
- `topo::SOMTopology{T}`: Topology to update
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)
- `bmus::Vector{Tuple{Int,Int}}`: BMU positions for each sample
- `σ::T`: Neighborhood width
"""
function update_weights_batch!(topo::SOMTopology{T}, X::AbstractMatrix{T},
                               bmus::Vector{Tuple{Int,Int}}, σ::T) where T
    # Auto-dispatch based on NaN presence
    if has_nan(X)
        _update_weights_batch_nan!(topo, X, bmus, σ)
    else
        _update_weights_batch_complete!(topo, X, bmus, σ)
    end
end

"""
    _update_weights_batch_complete!(topo, X, bmus, σ)

Fast weight update for complete data (no NaN values).
"""
function _update_weights_batch_complete!(topo::SOMTopology{T}, X::AbstractMatrix{T},
                                         bmus::Vector{Tuple{Int,Int}}, σ::T) where T
    n_samples = size(X, 2)
    input_dim = topo.input_dim

    for (pos, neuron) in topo.neurons
        numerator = zeros(T, input_dim)
        denominator = zero(T)

        @inbounds for i in 1:n_samples
            bmu_pos = bmus[i]
            dist = grid_distance(pos, bmu_pos)
            h = gaussian_neighborhood(dist, σ)

            for d in 1:input_dim
                numerator[d] += h * X[d, i]
            end
            denominator += h
        end

        if denominator > eps(T)
            @inbounds for d in 1:input_dim
                neuron.weights[d] = numerator[d] / denominator
            end
        end
    end
end

"""
    _update_weights_batch_nan!(topo, X, bmus, σ)

Weight update that handles NaN values with per-dimension accumulators.
Each dimension is updated independently based only on samples with valid values.
"""
function _update_weights_batch_nan!(topo::SOMTopology{T}, X::AbstractMatrix{T},
                                    bmus::Vector{Tuple{Int,Int}}, σ::T) where T
    n_samples = size(X, 2)
    input_dim = topo.input_dim

    for (pos, neuron) in topo.neurons
        numerator = zeros(T, input_dim)
        denominator = zeros(T, input_dim)  # Per-dimension denominator

        @inbounds for i in 1:n_samples
            bmu_pos = bmus[i]
            dist = grid_distance(pos, bmu_pos)
            h = gaussian_neighborhood(dist, σ)

            for d in 1:input_dim
                xval = X[d, i]
                if !isnan(xval)
                    numerator[d] += h * xval
                    denominator[d] += h
                end
            end
        end

        # Update each dimension independently
        @inbounds for d in 1:input_dim
            if denominator[d] > eps(T)
                neuron.weights[d] = numerator[d] / denominator[d]
            end
            # If no valid data for this dimension, weight unchanged
        end
    end
end

update_weights_batch!(som::DBGSOM{T}, X::AbstractMatrix{T},
                      bmus::Vector{Tuple{Int,Int}}, σ::T) where T =
    update_weights_batch!(som.topology, X, bmus, σ)

# ============================================================================
# Quantization Error
# ============================================================================

"""
    compute_quantization_error(topo, X) -> T

Compute mean quantization error over all samples.

QE = (1/N) Σᵢ ||xᵢ - w_bmuᵢ||
"""
function compute_quantization_error(topo::SOMTopology{T}, X::AbstractMatrix{T}) where T
    _, distances = find_bmus_with_distances(topo, X)
    return mean(distances)
end

compute_quantization_error(som::DBGSOM{T}, X::AbstractMatrix{T}) where T =
    compute_quantization_error(som.topology, X)

# ============================================================================
# Growth Threshold
# ============================================================================

"""
    compute_growth_threshold(som, X) -> T
    compute_growth_threshold(som) -> T

Compute growth threshold using the configured method.

# Methods
- `SpreadingFactor`: GT = -log(sf) × d (does not require data)
- `StatisticalError`: GT = λ × sqrt(Σ std_i²) (requires data)

# Arguments
- `som::DBGSOM{T}`: SOM with configured gt_method
- `X::AbstractMatrix{T}`: Input data (required for StatisticalError)

# Returns
Growth threshold value
"""
function compute_growth_threshold end

# Main dispatch - with data (works for both methods)
compute_growth_threshold(som::DBGSOM{T}, X::AbstractMatrix{T}) where T =
    _compute_gt(som.gt_method, som.topology.input_dim, X)

# Main dispatch - without data (only works for SpreadingFactor)
compute_growth_threshold(som::DBGSOM{T}) where T =
    _compute_gt(som.gt_method, som.topology.input_dim)

"""
    _compute_gt(method::SpreadingFactor, input_dim, [X]) -> T

Compute GT using spreading factor: GT = -log(sf) × d

Does not require training data.
"""
function _compute_gt(method::SpreadingFactor{T}, input_dim::Int) where T
    d = T(input_dim)
    return -log(method.sf) * d
end

# SpreadingFactor can ignore data if provided
_compute_gt(method::SpreadingFactor{T}, input_dim::Int, X::AbstractMatrix) where T =
    _compute_gt(method, input_dim)

"""
    _compute_gt(method::StatisticalError, input_dim, X) -> T

Compute GT using statistical error: GT = λ × sqrt(Σ std_i²)

Requires training data to compute per-dimension standard deviations.

# Formula
```
std_i = sqrt(sum((X[i,j] - mean_i)²) / (m-1))  for each dimension i
GT = λ × sqrt(Σ std_i²)
```

This is equivalent to λ × ||std_vector||₂ (L2 norm of std vector).
"""
function _compute_gt(method::StatisticalError{T}, input_dim::Int,
                     X::AbstractMatrix{T}) where T
    # Compute standard deviation for each dimension
    # std(X, dims=2) computes std across columns (samples) for each row (dimension)
    if any(isnan, X)
        stds = _compute_stds_nan_safe(X)
    else
        stds = vec(std(X, dims=2))  # Vector of length input_dim
    end

    # GT = λ × sqrt(Σ std_i²)
    # This is equivalent to λ × norm(stds)
    return method.lambda * sqrt(sum(stds .^ 2))
end

# StatisticalError REQUIRES data - error if called without it
function _compute_gt(method::StatisticalError{T}, input_dim::Int) where T
    error("StatisticalError method requires training data. Use compute_growth_threshold(som, X) instead.")
end

"""
    _compute_stds_nan_safe(X) -> Vector{T}

Compute per-dimension standard deviations, handling NaN values.
Uses only valid (non-NaN) values for each dimension independently.
"""
function _compute_stds_nan_safe(X::AbstractMatrix{T}) where T
    input_dim, n_samples = size(X)
    stds = zeros(T, input_dim)

    for d in 1:input_dim
        valid_vals = filter(!isnan, view(X, d, :))
        if length(valid_vals) > 1
            stds[d] = std(valid_vals)
        else
            stds[d] = zero(T)  # No variance if 0 or 1 valid values
        end
    end

    return stds
end

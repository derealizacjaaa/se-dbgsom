"""
NaN handling with missingness-weighted distances and neighbor-based imputation.

Replaces the simple "skip and scale" approach with:
1. Per-variable observation weights based on missingness rates
2. Distance computation weighted by variable reliability
3. Neighbor-based imputation for neuron dimensions with insufficient data
"""

# MissingnessInfo{T} struct is defined in core/types.jl

# ============================================================================
# Missingness Computation
# ============================================================================

"""
    compute_missingness_info(X) -> MissingnessInfo

Compute per-dimension missingness rates and observation weights from data.

Observation weight for dimension d is `(1 - rate_d)²`, which penalizes
high-missingness variables quadratically. A variable with 50% missingness
gets weight 0.25, while a fully observed variable gets weight 1.0.

# Arguments
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)
"""
function compute_missingness_info(X::AbstractMatrix{T}) where T<:AbstractFloat
    input_dim, n_samples = size(X)
    @assert input_dim > 0 "Data must have at least one dimension"
    @assert n_samples > 0 "Data must have at least one sample"
    rates = Vector{T}(undef, input_dim)

    @inbounds for d in 1:input_dim
        nan_count = 0
        for i in 1:n_samples
            if isnan(X[d, i])
                nan_count += 1
            end
        end
        rates[d] = T(nan_count) / T(n_samples)
    end

    # Quadratic penalty: (1 - rate)² so high-missingness dims contribute much less
    obs_weights = (one(T) .- rates) .^ 2

    return MissingnessInfo{T}(rates, obs_weights, n_samples)
end

# ============================================================================
# Missingness-Weighted Distance
# ============================================================================

"""
    euclidean_squared_nan_weighted(x, y, info) -> T

Squared Euclidean distance weighted by per-variable observation rates.

Each valid dimension contributes `obs_weight[d] * (x[d] - y[d])²`.
The result is normalized by `total_weight / used_weight` to remain
comparable across samples with different missingness patterns.

Variables with high missingness rates have lower observation weights,
so they contribute less to distance computation.

Returns `typemax(T)` if no valid dimensions exist.
"""
function euclidean_squared_nan_weighted(x::AbstractVector{T}, y::AbstractVector{T},
                                        info::MissingnessInfo{T}) where T
    dist = zero(T)
    used_weight = zero(T)
    total_weight = zero(T)

    @inbounds for i in eachindex(x, y)
        w = info.obs_weights[i]
        total_weight += w

        xi, yi = x[i], y[i]
        if !isnan(xi) && !isnan(yi) && w > zero(T)
            diff = xi - yi
            dist += w * diff * diff
            used_weight += w
        end
    end

    if used_weight > zero(T) && total_weight > zero(T)
        dist *= total_weight / used_weight
    else
        dist = typemax(T)
    end

    return dist
end

"""
    pairwise_distances_squared_nan_weighted(X, W, info) -> Matrix{T}

Compute pairwise squared distances using missingness-weighted metric.

# Arguments
- `X::Matrix{T}`: Input data (input_dim × n_samples)
- `W::Matrix{T}`: Weight vectors (input_dim × n_neurons)
- `info::MissingnessInfo{T}`: Missingness statistics
"""
function pairwise_distances_squared_nan_weighted(X::Matrix{T}, W::Matrix{T},
                                                  info::MissingnessInfo{T}) where T
    n_samples = size(X, 2)
    n_neurons = size(W, 2)
    D = Matrix{T}(undef, n_samples, n_neurons)

    @inbounds for j in 1:n_neurons
        w = view(W, :, j)
        for i in 1:n_samples
            x = view(X, :, i)
            D[i, j] = euclidean_squared_nan_weighted(x, w, info)
        end
    end

    return D
end

# ============================================================================
# Neighbor-Based Weight Imputation
# ============================================================================

"""
    impute_neuron_from_neighbors!(topo, pos, dim_obs_counts; min_obs=5.0, max_dist=2)

Impute poorly-observed weight dimensions from SOM grid neighbors.

For each dimension where `dim_obs_counts[d] < min_obs`, replaces the
neuron's weight with an inverse-distance-weighted average of neighboring
neurons' weights for that dimension.

This leverages the SOM's topological structure: nearby neurons should have
similar weights, so neighbors provide reliable estimates for dimensions
where the neuron itself received insufficient training data.

# Arguments
- `topo::SOMTopology{T}`: SOM topology
- `pos::Tuple{Int,Int}`: Position of neuron to impute
- `dim_obs_counts::Vector`: Effective observation count per dimension
- `min_obs::Real`: Minimum observations required (below triggers imputation)
- `max_dist::Int`: Maximum grid distance to search for neighbors
"""
function impute_neuron_from_neighbors!(topo::SOMTopology{T}, pos::Tuple{Int,Int},
                                       dim_obs_counts::AbstractVector;
                                       min_obs::Real=5.0,
                                       max_dist::Int=2) where T
    neuron = topo.neurons[pos]
    input_dim = topo.input_dim

    # Collect neighbors within max_dist
    neighbors = Tuple{Tuple{Int,Int}, Int}[]  # (position, distance)
    for (other_pos, _) in topo.neurons
        other_pos == pos && continue
        d = grid_distance(pos, other_pos)
        if d <= max_dist
            push!(neighbors, (other_pos, d))
        end
    end

    isempty(neighbors) && return nothing

    # Impute each under-observed dimension
    @inbounds for d in 1:input_dim
        if dim_obs_counts[d] < min_obs
            numerator = zero(T)
            denominator = zero(T)

            for (npos, gdist) in neighbors
                nb = topo.neurons[npos]
                w = one(T) / T(gdist)  # Inverse grid distance weight
                numerator += w * nb.weights[d]
                denominator += w
            end

            if denominator > eps(T)
                neuron.weights[d] = numerator / denominator
            end
        end
    end

    return nothing
end

# ============================================================================
# Batch Update with Imputation
# ============================================================================

"""
    update_weights_batch_nan_impute!(topo, X, bmus, σ, info) -> Dict

NaN-aware batch weight update with neighbor-based imputation.

Replaces `_update_weights_batch_nan!` with an improved version that:
1. Weights each dimension's contribution by its observation rate
2. Tracks per-neuron, per-dimension effective observation counts
3. Imputes poorly-observed dimensions from SOM grid neighbors

# Returns
Dict mapping neuron position => Vector of per-dimension observation counts
"""
function update_weights_batch_nan_impute!(topo::SOMTopology{T}, X::AbstractMatrix{T},
                                           bmus::Vector{Tuple{Int,Int}}, σ::T,
                                           info::MissingnessInfo{T}) where T
    n_samples = size(X, 2)
    input_dim = topo.input_dim

    # Track per-neuron, per-dimension observation counts
    obs_counts = Dict{Tuple{Int,Int}, Vector{T}}()

    for (pos, neuron) in topo.neurons
        numerator = zeros(T, input_dim)
        denominator = zeros(T, input_dim)

        @inbounds for i in 1:n_samples
            bmu_pos = bmus[i]
            dist = grid_distance(pos, bmu_pos)
            h = gaussian_neighborhood(dist, σ)

            for d in 1:input_dim
                xval = X[d, i]
                if !isnan(xval)
                    w = info.obs_weights[d]
                    numerator[d] += h * w * xval
                    denominator[d] += h * w
                end
            end
        end

        # Update weights where we have sufficient data
        @inbounds for d in 1:input_dim
            if denominator[d] > eps(T)
                neuron.weights[d] = numerator[d] / denominator[d]
            end
        end

        obs_counts[pos] = denominator
    end

    # Impute poorly-observed dimensions from neighbors.
    # Threshold combines absolute minimum (at least 1.0 effective observation)
    # with relative check (10% of the neuron's best-observed dimension).
    # This ensures imputation fires both for truly sparse neurons and for
    # dimensions that are disproportionately under-observed relative to others.
    for (pos, counts) in obs_counts
        max_count = maximum(counts)
        relative_threshold = max_count > zero(T) ? max_count * T(0.1) : zero(T)
        threshold = max(T(1.0), relative_threshold)
        impute_neuron_from_neighbors!(topo, pos, counts; min_obs=threshold)
    end

    return obs_counts
end

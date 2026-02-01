"""
Best Matching Unit (BMU) finding for BasicDBGSOM.
"""

# ============================================================================
# Single Sample BMU
# ============================================================================

"""
    find_bmu(topo, x) -> (position, distance)

Find the Best Matching Unit for input vector x.
Automatically handles NaN values in input by using scaled partial distances.

# Returns
- `position::Tuple{Int,Int}`: Grid position of BMU
- `distance::T`: Euclidean distance to BMU
"""
function find_bmu(topo::SOMTopology{T}, x::AbstractVector{T}) where T
    best_pos = nothing
    best_dist_sq = typemax(T)

    # Check once if input has NaN
    x_has_nan = any(isnan, x)

    for (pos, neuron) in topo.neurons
        dist_sq = if x_has_nan
            euclidean_squared_nan(neuron.weights, x)
        else
            euclidean_squared(neuron.weights, x)
        end
        if dist_sq < best_dist_sq
            best_dist_sq = dist_sq
            best_pos = pos
        end
    end

    return best_pos, sqrt(best_dist_sq)
end

find_bmu(som::DBGSOM{T}, x::AbstractVector{T}) where T = find_bmu(som.topology, x)

"""
    find_bmu_idx(positions, D_row) -> Int

Find BMU index from precomputed distance row.
"""
function find_bmu_idx(D_row::AbstractVector{T}) where T
    best_idx = 1
    best_dist = D_row[1]

    @inbounds for i in 2:length(D_row)
        if D_row[i] < best_dist
            best_dist = D_row[i]
            best_idx = i
        end
    end

    return best_idx
end

# ============================================================================
# Batch BMU Finding
# ============================================================================

"""
    find_bmus(topo, X) -> Vector{Tuple{Int,Int}}

Find BMUs for all samples in X.
Automatically handles NaN values by using scaled partial distances.

# Arguments
- `topo::SOMTopology{T}`: Topology with neurons
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)

# Returns
Vector of BMU positions for each sample.
"""
function find_bmus(topo::SOMTopology{T}, X::AbstractMatrix{T}) where T
    n_samples = size(X, 2)
    pos_list, W = get_positions_and_weights(topo)

    # Compute all pairwise distances (auto-dispatches based on NaN presence)
    D = pairwise_distances_squared_auto(Matrix(X), W)

    # Find argmin for each sample
    bmus = Vector{Tuple{Int,Int}}(undef, n_samples)

    @inbounds for i in 1:n_samples
        idx = find_bmu_idx(view(D, i, :))
        bmus[i] = pos_list[idx]
    end

    return bmus
end

find_bmus(som::DBGSOM{T}, X::AbstractMatrix{T}) where T = find_bmus(som.topology, X)

"""
    find_bmus_with_distances(topo, X) -> (Vector{Tuple{Int,Int}}, Vector{T})

Find BMUs and their distances for all samples.
Automatically handles NaN values by using scaled partial distances.

# Returns
- `bmus::Vector{Tuple{Int,Int}}`: BMU positions
- `distances::Vector{T}`: Euclidean distances to BMUs
"""
function find_bmus_with_distances(topo::SOMTopology{T}, X::AbstractMatrix{T}) where T
    n_samples = size(X, 2)
    pos_list, W = get_positions_and_weights(topo)

    # Compute all pairwise distances (auto-dispatches based on NaN presence)
    D = pairwise_distances_squared_auto(Matrix(X), W)

    bmus = Vector{Tuple{Int,Int}}(undef, n_samples)
    distances = Vector{T}(undef, n_samples)

    @inbounds for i in 1:n_samples
        idx = find_bmu_idx(view(D, i, :))
        bmus[i] = pos_list[idx]
        distances[i] = sqrt(D[i, idx])
    end

    return bmus, distances
end

find_bmus_with_distances(som::DBGSOM{T}, X::AbstractMatrix{T}) where T =
    find_bmus_with_distances(som.topology, X)

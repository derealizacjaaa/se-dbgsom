"""
Growth and pruning functions for BasicDBGSOM.
"""

# ============================================================================
# Error Accumulation
# ============================================================================

"""
    reset_errors!(topo)

Reset accumulated errors and hit counts for all neurons.
"""
function reset_errors!(topo::SOMTopology{T}) where T
    for neuron in values(topo.neurons)
        neuron.error = zero(T)
        neuron.hits = 0
    end
end

reset_errors!(som::DBGSOM) = reset_errors!(som.topology)

"""
    accumulate_errors!(topo, X, bmus)

Accumulate quantization errors for each neuron based on BMU assignments.
Handles NaN values by using scaled partial distances.

# Arguments
- `topo::SOMTopology{T}`: Topology to update
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)
- `bmus::Vector{Tuple{Int,Int}}`: BMU positions for each sample
"""
function accumulate_errors!(topo::SOMTopology{T}, X::AbstractMatrix{T},
                            bmus::Vector{Tuple{Int,Int}}) where T
    n_samples = size(X, 2)

    @inbounds for i in 1:n_samples
        pos = bmus[i]
        neuron = topo.neurons[pos]
        x = view(X, :, i)

        # Compute distance to sample (NaN-aware)
        dist = if any(isnan, x)
            euclidean_nan(neuron.weights, x)
        else
            euclidean(neuron.weights, x)
        end

        # Only accumulate if distance is valid
        if !isnan(dist) && dist < typemax(T)
            neuron.error += dist
            neuron.hits += 1
        end
    end
end

accumulate_errors!(som::DBGSOM{T}, X::AbstractMatrix{T},
                   bmus::Vector{Tuple{Int,Int}}) where T =
    accumulate_errors!(som.topology, X, bmus)

"""
    accumulate_errors!(topo, X, bmus, miss_info)

Accumulate quantization errors using missingness-weighted distances when
`miss_info` is provided. Falls back to standard distances when `miss_info`
is `nothing`.
"""
function accumulate_errors!(topo::SOMTopology{T}, X::AbstractMatrix{T},
                            bmus::Vector{Tuple{Int,Int}},
                            miss_info::Union{Nothing, MissingnessInfo{T}}) where T
    if miss_info === nothing
        return accumulate_errors!(topo, X, bmus)
    end

    n_samples = size(X, 2)

    @inbounds for i in 1:n_samples
        pos = bmus[i]
        neuron = topo.neurons[pos]
        x = view(X, :, i)

        # Compute distance using missingness-weighted metric
        dist_sq = if any(isnan, x)
            euclidean_squared_nan_weighted(neuron.weights, x, miss_info)
        else
            euclidean_squared(neuron.weights, x)
        end

        dist = sqrt(dist_sq)

        # Only accumulate if distance is valid
        if !isnan(dist) && dist < typemax(T)
            neuron.error += dist
            neuron.hits += 1
        end
    end
end

accumulate_errors!(som::DBGSOM{T}, X::AbstractMatrix{T},
                   bmus::Vector{Tuple{Int,Int}},
                   miss_info::Union{Nothing, MissingnessInfo{T}}) where T =
    accumulate_errors!(som.topology, X, bmus, miss_info)

# ============================================================================
# Error Distribution
# ============================================================================

"""
    distribute_errors!(topo, gt)

Distribute errors from non-boundary neurons to their boundary neighbors.

For each non-boundary neuron where E_i ≥ GT:
- Each boundary neighbor receives error/12 (hexagonal grid has up to 12 second-order neighbors)
- The non-boundary neuron retains error/2

Adapted from Vasighi & Amini (2017) for hexagonal grid topology.
"""
function distribute_errors!(topo::SOMTopology{T}, gt::T) where T
    # Collect non-boundary neurons with high error
    for (pos, neuron) in topo.neurons
        if neuron.error >= gt && !is_boundary(topo, pos)
            # Find boundary neighbors (direct neighbors that are on boundary)
            boundary_neighbors = Tuple{Int,Int}[]
            for neighbor_pos in get_neighbors(pos)
                if has_neuron(topo, neighbor_pos) && is_boundary(topo, neighbor_pos)
                    push!(boundary_neighbors, neighbor_pos)
                end
            end

            # Distribute error/12 to each boundary neighbor, retain error/2
            # (12 accounts for hexagonal second-order neighborhood size)
            if !isempty(boundary_neighbors)
                error_share = neuron.error / T(12)
                for neighbor_pos in boundary_neighbors
                    topo.neurons[neighbor_pos].error += error_share
                end
                neuron.error = neuron.error / T(2)
            end
        end
    end
end

distribute_errors!(som::DBGSOM{T}, gt::T) where T =
    distribute_errors!(som.topology, gt)

# ============================================================================
# Growth Candidates
# ============================================================================

"""
    find_growth_candidates(topo, gt) -> Vector{Tuple{Int,Int}}

Find boundary neurons with error exceeding growth threshold.

# Arguments
- `topo::SOMTopology{T}`: Topology to search
- `gt::T`: Growth threshold

# Returns
Positions sorted by error (highest first)
"""
function find_growth_candidates(topo::SOMTopology{T}, gt::T) where T
    candidates = Tuple{Int,Int}[]

    for (pos, neuron) in topo.neurons
        if neuron.error > gt && is_boundary(topo, pos)
            push!(candidates, pos)
        end
    end

    # Sort by error (highest first)
    sort!(candidates, by=pos -> topo.neurons[pos].error, rev=true)

    return candidates
end

find_growth_candidates(som::DBGSOM{T}, gt::T) where T =
    find_growth_candidates(som.topology, gt)

# ============================================================================
# Directed Growth (Vasighi & Amini 2017)
# ============================================================================

"""
    select_growth_direction(topo, boundary_pos) -> (new_pos, weights) or nothing

Select the best growth direction from a boundary neuron based on error distribution
of neighbors. This implements the "directed" part of DBGSOM for hexagonal grids.

The algorithm considers:
- 1st-order neighbors (distance=1): direct neighbors (up to 6 in hex grid)
- 2nd-order neighbors (distance=2): neighbors of neighbors
- Growth direction is biased toward regions with higher accumulated error

Adapted from Vasighi & Amini (2017) for hexagonal grid topology.

# Returns
Tuple of (new_position, weight_vector) or nothing if no growth possible
"""
function select_growth_direction(topo::SOMTopology{T}, boundary_pos::Tuple{Int,Int}) where T
    parent = topo.neurons[boundary_pos]
    allowed = get_empty_neighbors(topo, boundary_pos)

    if isempty(allowed)
        return nothing
    end

    # Find 1st-order neighbors (distance=1) and 2nd-order neighbors (distance=2)
    nbr1 = get_neighbors_at_distance(topo, boundary_pos, 1)  # 0-5 occupied (out of 6)
    nbr2 = get_neighbors_at_distance(topo, boundary_pos, 2)  # 2nd order neighbors

    # Sort neighbors by error (highest first)
    sort!(nbr1, by=p -> topo.neurons[p].error, rev=true)
    sort!(nbr2, by=p -> topo.neurons[p].error, rev=true)

    n_nbr1 = length(nbr1)

    # Hex cases: boundary neuron has 0-5 occupied neighbors (6 - n_allowed)
    if n_nbr1 == 0
        # Isolated neuron - grow in random direction
        return _grow_isolated_hex(topo, boundary_pos, allowed)
    elseif n_nbr1 <= 2
        # Few neighbors (1-2): grow away from neighbors toward high-error region
        return _grow_few_neighbors_hex(topo, boundary_pos, allowed, nbr1, nbr2)
    elseif n_nbr1 <= 4
        # Medium neighbors (3-4): grow toward highest error direction
        return _grow_medium_neighbors_hex(topo, boundary_pos, allowed, nbr1, nbr2)
    else
        # Many neighbors (5): only one slot available
        return _grow_many_neighbors_hex(topo, boundary_pos, allowed, nbr1, nbr2)
    end
end

"""
    _grow_isolated_hex(topo, bo_pos, allowed)

Growth when boundary neuron has no neighbors (isolated).
Copies parent weights with small noise.
"""
function _grow_isolated_hex(topo::SOMTopology{T}, bo_pos::Tuple{Int,Int},
                            allowed::Vector{Tuple{Int,Int}}) where T
    parent = topo.neurons[bo_pos]
    new_pos = allowed[1]
    weights = copy(parent.weights)
    weights .+= randn(T, length(weights)) .* T(0.01)
    return (new_pos, weights)
end

"""
    _grow_few_neighbors_hex(topo, bo_pos, allowed, nbr1, nbr2)

Growth when boundary neuron has 1-2 direct neighbors (4-5 empty slots).
Prefers growing toward highest-error 2nd-order neighbor, or away from existing neighbors.
"""
function _grow_few_neighbors_hex(topo::SOMTopology{T}, bo_pos::Tuple{Int,Int},
                                  allowed::Vector{Tuple{Int,Int}},
                                  nbr1::Vector{Tuple{Int,Int}},
                                  nbr2::Vector{Tuple{Int,Int}}) where T
    parent = topo.neurons[bo_pos]

    # Find allowed positions adjacent to high-error nbr2
    sel_nbr2 = filter(p -> any(grid_distance(p, a) == 1 for a in allowed), nbr2)

    if !isempty(sel_nbr2)
        sort!(sel_nbr2, by=p -> topo.neurons[p].error, rev=true)
        best_nbr2 = topo.neurons[sel_nbr2[1]]
        best_nbr2_pos = sel_nbr2[1]
        # Find allowed position closest to best_nbr2
        new_pos = argmin(a -> grid_distance(a, best_nbr2_pos), allowed)
        weights = (parent.weights .+ best_nbr2.weights) ./ T(2)
    else
        # Grow away from first neighbor using reflection
        nbr = topo.neurons[nbr1[1]]
        nbr1_pos = nbr1[1]
        # Pick position farthest from nbr1[1]
        new_pos = argmax(a -> grid_distance(a, nbr1_pos), allowed)
        weights = 2 .* parent.weights .- nbr.weights
    end

    return (new_pos, weights)
end

"""
    _grow_medium_neighbors_hex(topo, bo_pos, allowed, nbr1, nbr2)

Growth when boundary neuron has 3-4 direct neighbors (2-3 empty slots).
Grows toward the direction with highest accumulated error.
"""
function _grow_medium_neighbors_hex(topo::SOMTopology{T}, bo_pos::Tuple{Int,Int},
                                     allowed::Vector{Tuple{Int,Int}},
                                     nbr1::Vector{Tuple{Int,Int}},
                                     nbr2::Vector{Tuple{Int,Int}}) where T
    parent = topo.neurons[bo_pos]
    max_err_nbr_pos = nbr1[1]  # Highest error neighbor (sorted)
    min_err_nbr = topo.neurons[nbr1[end]]  # Lowest error neighbor

    # Pick allowed position closest to highest-error neighbor
    new_pos = argmin(a -> grid_distance(a, max_err_nbr_pos), allowed)

    # Use reflection from lowest-error neighbor
    weights = 2 .* parent.weights .- min_err_nbr.weights

    return (new_pos, weights)
end

"""
    _grow_many_neighbors_hex(topo, bo_pos, allowed, nbr1, nbr2)

Growth when boundary neuron has 5 direct neighbors (1 empty slot).
Only one direction possible, uses opposite neighbor for reflection.
"""
function _grow_many_neighbors_hex(topo::SOMTopology{T}, bo_pos::Tuple{Int,Int},
                                   allowed::Vector{Tuple{Int,Int}},
                                   nbr1::Vector{Tuple{Int,Int}},
                                   nbr2::Vector{Tuple{Int,Int}}) where T
    parent = topo.neurons[bo_pos]
    new_pos = allowed[1]

    # Find opposite neighbor (distance 2 from new_pos)
    opposite = filter(n -> grid_distance(n, new_pos) == 2, nbr1)

    if !isempty(opposite)
        opp_nbr = topo.neurons[opposite[1]]
        weights = 2 .* parent.weights .- opp_nbr.weights
    else
        # No clear opposite - average reflection from all neighbors
        avg_weights = mean([topo.neurons[n].weights for n in nbr1])
        weights = 2 .* parent.weights .- avg_weights
    end

    return (new_pos, weights)
end

"""
    insert_neuron_directed!(topo, pos, weights)

Insert a new neuron with pre-computed weights at the specified position.
"""
function insert_neuron_directed!(topo::SOMTopology{T}, pos::Tuple{Int,Int},
                                  weights::Vector{T}) where T
    @assert !has_neuron(topo, pos) "Position $pos already occupied"
    topo.neurons[pos] = Neuron{T}(weights)
end

insert_neuron_directed!(som::DBGSOM{T}, pos::Tuple{Int,Int}, weights::Vector{T}) where T =
    insert_neuron_directed!(som.topology, pos, weights)

# ============================================================================
# Legacy Neuron Insertion (kept for compatibility)
# ============================================================================

"""
    insert_neuron!(topo, pos, parent_pos)

Insert a new neuron at pos, initialized from parent neuron using reflection formula.
This is the simpler version; prefer using select_growth_direction + insert_neuron_directed!
for the full directed growth algorithm.
"""
function insert_neuron!(topo::SOMTopology{T}, pos::Tuple{Int,Int},
                        parent_pos::Tuple{Int,Int}) where T
    @assert !has_neuron(topo, pos) "Position $pos already occupied"
    @assert has_neuron(topo, parent_pos) "Parent position $parent_pos not found"

    parent = topo.neurons[parent_pos]
    occupied_neighbors = get_occupied_neighbors(topo, parent_pos)

    if !isempty(occupied_neighbors)
        neighbor_pos = occupied_neighbors[1]
        neighbor = topo.neurons[neighbor_pos]
        weights = 2 .* parent.weights .- neighbor.weights
    else
        weights = copy(parent.weights)
        weights .+= randn(T, length(weights)) .* T(0.01)
    end

    topo.neurons[pos] = Neuron{T}(weights)
end

insert_neuron!(som::DBGSOM{T}, pos::Tuple{Int,Int}, parent_pos::Tuple{Int,Int}) where T =
    insert_neuron!(som.topology, pos, parent_pos)

# ============================================================================
# Growth Step
# ============================================================================

"""
    grow!(som, gt) -> Int

Grow neurons from high-error boundary neurons using directed growth.

Uses the full DBGSOM directed growth algorithm:
- Considers 1st and 2nd-order neighbors
- Growth direction biased toward high-error regions
- Weight initialization uses reflection with neighbor error weighting

Reference: Vasighi & Amini (2017)

# Arguments
- `som::DBGSOM{T}`: SOM to grow
- `gt::T`: Growth threshold

# Returns
Number of neurons added
"""
function grow!(som::DBGSOM{T}, gt::T) where T
    candidates = find_growth_candidates(som, gt)
    n_added = 0

    for pos in candidates
        if n_neurons(som) >= som.max_neurons
            break
        end

        # Use directed growth to select position and compute weights
        result = select_growth_direction(som.topology, pos)

        if result !== nothing
            new_pos, weights = result
            insert_neuron_directed!(som, new_pos, weights)

            # Reset parent error after growing
            som.topology.neurons[pos].error = zero(T)

            n_added += 1
        end
    end

    return n_added
end

# ============================================================================
# Pruning
# ============================================================================

"""
    find_dead_neurons(topo; min_hits=1) -> Vector{Tuple{Int,Int}}

Find neurons that have been activated fewer than min_hits times.
"""
function find_dead_neurons(topo::SOMTopology; min_hits::Int=1)
    dead = Tuple{Int,Int}[]

    for (pos, neuron) in topo.neurons
        if neuron.hits < min_hits
            push!(dead, pos)
        end
    end

    return dead
end

find_dead_neurons(som::DBGSOM; min_hits::Int=1) =
    find_dead_neurons(som.topology; min_hits=min_hits)

"""
    prune_dead_neurons!(som; min_hits=1, min_neurons=4) -> Int

Remove neurons that have been activated fewer than min_hits times.
Keeps at least min_neurons in the map.

# Returns
Number of neurons removed
"""
function prune_dead_neurons!(som::DBGSOM; min_hits::Int=1, min_neurons::Int=4)
    dead = find_dead_neurons(som; min_hits=min_hits)
    n_removed = 0

    for pos in dead
        if n_neurons(som) > min_neurons
            remove_neuron!(som, pos)
            n_removed += 1
        end
    end

    return n_removed
end

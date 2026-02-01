"""
Neighborhood functions for BasicDBGSOM.
"""

# ============================================================================
# Gaussian Neighborhood
# ============================================================================

"""
    gaussian_neighborhood(dist, σ) -> T

Gaussian neighborhood function: h(d, σ) = exp(-d² / (2σ²))

# Arguments
- `dist`: Distance between neurons on the grid
- `σ`: Neighborhood width parameter
"""
function gaussian_neighborhood(dist::Real, σ::T) where T<:AbstractFloat
    return exp(-dist^2 / (2 * σ^2))
end

"""
    compute_neighborhood_matrix(positions, σ) -> Dict

Compute pairwise neighborhood values for all neuron positions.

# Returns
Dict mapping (pos1, pos2) => neighborhood_value
"""
function compute_neighborhood_matrix(positions::Vector{Tuple{Int,Int}}, σ::T) where T
    H = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, T}()

    for pos1 in positions
        for pos2 in positions
            dist = grid_distance(pos1, pos2)
            H[(pos1, pos2)] = gaussian_neighborhood(dist, σ)
        end
    end

    return H
end

# ============================================================================
# Sigma Computation
# ============================================================================

"""
    compute_sigma(topo) -> T

Compute neighborhood width from current grid extent.
σ = grid_extent / 2
"""
function compute_sigma(topo::SOMTopology{T}) where T
    extent = grid_extent(topo)
    return T(max(extent, 1) / 2)
end

compute_sigma(som::DBGSOM) = compute_sigma(som.topology)

"""
    decay_sigma(epoch, n_epochs, σ_start, σ_end) -> T

Linear decay of sigma from σ_start to σ_end over n_epochs.
"""
function decay_sigma(epoch::Int, n_epochs::Int, σ_start::T, σ_end::T) where T
    if n_epochs <= 1
        return σ_start
    end

    progress = (epoch - 1) / (n_epochs - 1)
    return σ_start - (σ_start - σ_end) * progress
end

"""
    compute_sigma_range(topo) -> (σ_start, σ_end)

Compute default sigma start and end values from topology.
"""
function compute_sigma_range(topo::SOMTopology{T}) where T
    σ_start = compute_sigma(topo)
    σ_end = max(σ_start * T(0.2), T(0.5))
    return σ_start, σ_end
end

compute_sigma_range(som::DBGSOM) = compute_sigma_range(som.topology)

"""
Grid topology functions for BasicDBGSOM.
"""

# ============================================================================
# Grid Navigation
# ============================================================================

"""
    get_neighbors(pos) -> Vector{Tuple{Int,Int}}

Get 6-connected neighbor positions for hexagonal grid (odd-r offset coordinates).

In odd-r offset, odd rows are shifted right by 0.5. This affects which cells
are considered neighbors depending on the row parity.
"""
function get_neighbors(pos::Tuple{Int,Int})
    x, y = pos
    if y % 2 == 0  # Even row
        return [
            (x+1, y),    # East
            (x,   y-1),  # North-East
            (x-1, y-1),  # North-West
            (x-1, y),    # West
            (x-1, y+1),  # South-West
            (x,   y+1)   # South-East
        ]
    else  # Odd row (shifted right)
        return [
            (x+1, y),    # East
            (x+1, y-1),  # North-East
            (x,   y-1),  # North-West
            (x-1, y),    # West
            (x,   y+1),  # South-West
            (x+1, y+1)   # South-East
        ]
    end
end

"""
    grid_distance(pos1, pos2) -> Int

Hexagonal distance between two grid positions using cube coordinate conversion.

For odd-r offset coordinates, we convert to cube coordinates and compute
the maximum of the three axis differences.
"""
function grid_distance(pos1::Tuple{Int,Int}, pos2::Tuple{Int,Int})
    # Convert odd-r offset to cube coordinates
    function to_cube(x, y)
        cx = x - (y - (y & 1)) ÷ 2
        cz = y
        cy = -cx - cz
        return (cx, cy, cz)
    end
    c1 = to_cube(pos1...)
    c2 = to_cube(pos2...)
    return max(abs(c1[1] - c2[1]), abs(c1[2] - c2[2]), abs(c1[3] - c2[3]))
end

"""
    euclidean_grid_distance(pos1, pos2) -> Float64

Euclidean distance between two grid positions.
"""
euclidean_grid_distance(pos1::Tuple{Int,Int}, pos2::Tuple{Int,Int}) =
    sqrt((pos1[1] - pos2[1])^2 + (pos1[2] - pos2[2])^2)

# ============================================================================
# Topology Queries
# ============================================================================

"""
    is_boundary(topo, pos) -> Bool

Check if position is on the boundary (has empty neighbor slots).
"""
function is_boundary(topo::SOMTopology, pos::Tuple{Int,Int})
    for neighbor_pos in get_neighbors(pos)
        if !has_neuron(topo, neighbor_pos)
            return true
        end
    end
    return false
end

is_boundary(som::DBGSOM, pos::Tuple{Int,Int}) = is_boundary(som.topology, pos)

"""
    get_empty_neighbors(topo, pos) -> Vector{Tuple{Int,Int}}

Get positions of empty neighbor slots.
"""
function get_empty_neighbors(topo::SOMTopology, pos::Tuple{Int,Int})
    empty = Tuple{Int,Int}[]
    for neighbor_pos in get_neighbors(pos)
        if !has_neuron(topo, neighbor_pos)
            push!(empty, neighbor_pos)
        end
    end
    return empty
end

get_empty_neighbors(som::DBGSOM, pos::Tuple{Int,Int}) =
    get_empty_neighbors(som.topology, pos)

"""
    get_occupied_neighbors(topo, pos) -> Vector{Tuple{Int,Int}}

Get positions of occupied neighbor slots.
"""
function get_occupied_neighbors(topo::SOMTopology, pos::Tuple{Int,Int})
    occupied = Tuple{Int,Int}[]
    for neighbor_pos in get_neighbors(pos)
        if has_neuron(topo, neighbor_pos)
            push!(occupied, neighbor_pos)
        end
    end
    return occupied
end

get_occupied_neighbors(som::DBGSOM, pos::Tuple{Int,Int}) =
    get_occupied_neighbors(som.topology, pos)

"""
    get_neighbors_at_distance(topo, pos, dist) -> Vector{Tuple{Int,Int}}

Get occupied neighbor positions at exactly the specified grid distance.
Distance 1 = direct neighbors, Distance 2 = 2nd-order neighbors.
"""
function get_neighbors_at_distance(topo::SOMTopology, pos::Tuple{Int,Int}, dist::Int)
    neighbors = Tuple{Int,Int}[]
    for (other_pos, _) in topo.neurons
        if grid_distance(pos, other_pos) == dist
            push!(neighbors, other_pos)
        end
    end
    return neighbors
end

get_neighbors_at_distance(som::DBGSOM, pos::Tuple{Int,Int}, dist::Int) =
    get_neighbors_at_distance(som.topology, pos, dist)

# ============================================================================
# Grid Bounds
# ============================================================================

"""
    grid_bounds(topo) -> ((x_min, x_max), (y_min, y_max))

Get the bounding box of the current grid.
"""
function grid_bounds(topo::SOMTopology)
    if isempty(topo.neurons)
        return ((0, 0), (0, 0))
    end

    pos_list = collect(positions(topo))
    x_min = minimum(p[1] for p in pos_list)
    x_max = maximum(p[1] for p in pos_list)
    y_min = minimum(p[2] for p in pos_list)
    y_max = maximum(p[2] for p in pos_list)

    return ((x_min, x_max), (y_min, y_max))
end

grid_bounds(som::DBGSOM) = grid_bounds(som.topology)

"""
    grid_extent(topo) -> Int

Get the maximum extent of the grid (max of width and height).
"""
function grid_extent(topo::SOMTopology)
    ((x_min, x_max), (y_min, y_max)) = grid_bounds(topo)
    return max(x_max - x_min + 1, y_max - y_min + 1)
end

grid_extent(som::DBGSOM) = grid_extent(som.topology)

# ============================================================================
# Neuron Management
# ============================================================================

"""
    add_neuron!(topo, pos, weights)

Add a neuron at the specified position.
"""
function add_neuron!(topo::SOMTopology{T}, pos::Tuple{Int,Int},
                     weights::Vector{T}) where T
    @assert !has_neuron(topo, pos) "Position $pos already occupied"
    @assert length(weights) == topo.input_dim "Weight dimension mismatch"

    topo.neurons[pos] = Neuron{T}(weights)
end

add_neuron!(som::DBGSOM{T}, pos::Tuple{Int,Int}, weights::Vector{T}) where T =
    add_neuron!(som.topology, pos, weights)

"""
    remove_neuron!(topo, pos)

Remove the neuron at the specified position.
"""
function remove_neuron!(topo::SOMTopology, pos::Tuple{Int,Int})
    @assert has_neuron(topo, pos) "No neuron at position $pos"
    delete!(topo.neurons, pos)
end

remove_neuron!(som::DBGSOM, pos::Tuple{Int,Int}) = remove_neuron!(som.topology, pos)

# ============================================================================
# Weight Access
# ============================================================================

"""
    get_weights(topo) -> Matrix{T}

Get all neuron weights as input_dim × n_neurons matrix.
"""
function get_weights(topo::SOMTopology{T}) where T
    n = n_neurons(topo)
    W = Matrix{T}(undef, topo.input_dim, n)

    for (i, neuron) in enumerate(values(topo.neurons))
        W[:, i] = neuron.weights
    end

    return W
end

get_weights(som::DBGSOM) = get_weights(som.topology)

"""
    get_positions_and_weights(topo) -> (Vector{Tuple{Int,Int}}, Matrix{T})

Get positions and corresponding weights.
"""
function get_positions_and_weights(topo::SOMTopology{T}) where T
    pos_list = collect(positions(topo))
    n = length(pos_list)
    W = Matrix{T}(undef, topo.input_dim, n)

    for (i, pos) in enumerate(pos_list)
        W[:, i] = topo.neurons[pos].weights
    end

    return pos_list, W
end

get_positions_and_weights(som::DBGSOM) = get_positions_and_weights(som.topology)

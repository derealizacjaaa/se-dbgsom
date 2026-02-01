"""
Visualization data extraction for BasicDBGSOM.

Provides functions to extract data for plotting with external libraries
(Plots.jl, Makie.jl, etc.) without depending on them directly.
"""

# ============================================================================
# Plotting Data Structures
# ============================================================================

"""
    PlottingData{T}

Container for all visualization data from a trained DBGSOM.

# Fields
- `positions::Vector{Tuple{Int,Int}}`: Neuron grid positions
- `weights::Matrix{T}`: Weight vectors (input_dim × n_neurons)
- `umatrix::Dict{Tuple{Int,Int}, T}`: U-matrix values per position
- `hit_counts::Dict{Tuple{Int,Int}, Int}`: Hit counts per position
- `edges::Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}`: Topology edges
- `bounds::Tuple{Tuple{Int,Int}, Tuple{Int,Int}}`: Grid bounds ((x_min, x_max), (y_min, y_max))
"""
struct PlottingData{T<:AbstractFloat}
    positions::Vector{Tuple{Int,Int}}
    weights::Matrix{T}
    umatrix::Dict{Tuple{Int,Int}, T}
    hit_counts::Dict{Tuple{Int,Int}, Int}
    edges::Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}
    bounds::Tuple{Tuple{Int,Int}, Tuple{Int,Int}}
end

# ============================================================================
# U-Matrix
# ============================================================================

"""
    compute_umatrix(som) -> Dict{Tuple{Int,Int}, T}

Compute the U-matrix (unified distance matrix) using standard procedure.

The U-matrix shows the average distance between each neuron and its neighbors.
High values indicate cluster boundaries, low values indicate cluster interiors.

# Algorithm (Standard Procedure)
For each node n:
1. Retrieve neighbors N(n) from topology
2. If N(n) is empty, set U_n = 0
3. Else compute average Euclidean distance to all neighbors:
   U_n = sum(dist(w_n, w_m) for m in N(n)) / |N(n)|

# Arguments
- `som`: Trained DBGSOM

# Returns
Dict mapping position => average neighbor distance value
"""
compute_umatrix(som::DBGSOM) = compute_umatrix_from_topology(som.topology)

compute_umatrix(topo::SOMTopology{T}) where T = compute_umatrix_from_topology(topo)

function compute_umatrix_from_topology(topo::SOMTopology{T}) where T
    umatrix = Dict{Tuple{Int,Int}, T}()

    for (pos, neuron) in topo.neurons
        neighbor_distances = T[]

        for neighbor_pos in get_neighbors(pos)
            if has_neuron(topo, neighbor_pos)
                neighbor = get_neuron(topo, neighbor_pos)
                dist = euclidean(neuron.weights, neighbor.weights)
                push!(neighbor_distances, dist)
            end
        end

        if isempty(neighbor_distances)
            umatrix[pos] = zero(T)
        else
            umatrix[pos] = mean(neighbor_distances)
        end
    end

    return umatrix
end

"""
    compute_expanded_umatrix(som; log=false) -> (neuron_umat, edge_umat)

Compute expanded U-matrix with per-edge distances for hexagonal visualization.

The expanded U-matrix provides both neuron-level (mean) and edge-level (pairwise)
distance information, enabling the classical detailed U-matrix visualization
where intermediate cells between neurons show individual pairwise distances.

# Arguments
- `som::DBGSOM{T}`: Trained DBGSOM
- `log::Bool=false`: If true, apply log transform to distances

# Returns
- `neuron_umat::Dict{Tuple{Int,Int}, T}`: Standard U-matrix (mean neighbor distance per neuron)
- `edge_umat::Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, T}`: Per-edge distances between adjacent neurons

The edge dict is keyed by sorted position pairs (pos1, pos2) where pos1 < pos2 to avoid duplicates.

# Example
```julia
neuron_umat, edge_umat = compute_expanded_umatrix(som)

# neuron_umat[(x,y)] = mean distance to neighbors at position (x,y)
# edge_umat[((x1,y1), (x2,y2))] = distance between adjacent neurons
```
"""
function compute_expanded_umatrix(som::DBGSOM{T}; log::Bool=false) where T
    neuron_umat = compute_umatrix(som)
    edge_umat = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, T}()

    # Compute per-edge distances
    for (pos, neuron) in som.topology.neurons
        for neighbor_pos in get_neighbors(pos)
            if has_neuron(som, neighbor_pos)
                # Use canonical ordering to avoid duplicates
                edge_key = pos < neighbor_pos ? (pos, neighbor_pos) : (neighbor_pos, pos)
                if !haskey(edge_umat, edge_key)
                    neighbor = get_neuron(som, neighbor_pos)
                    dist = euclidean(neuron.weights, neighbor.weights)
                    edge_umat[edge_key] = log ? Base.log(dist + eps(T)) : dist
                end
            end
        end
    end

    if log
        for (pos, val) in neuron_umat
            neuron_umat[pos] = Base.log(val + eps(T))
        end
    end

    return neuron_umat, edge_umat
end

"""
    compute_ustar_matrix(som, X) -> Dict{Tuple{Int,Int}, T}

Compute U* matrix combining U-matrix with density (hit counts).

Formula: U*(i) = U(i) * (1 - normalized_density(i))

This reduces U-values for neurons with high hit counts, making dense regions
(cluster interiors) have lower values than sparse boundary regions.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data (input_dim × n_samples)

# Returns
Dict mapping position => U* value
"""
function compute_ustar_matrix(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    umat = compute_umatrix(som)
    hits = compute_hit_counts(som, X)

    if isempty(umat)
        return Dict{Tuple{Int,Int}, T}()
    end

    # Find max hits for normalization
    max_hits = maximum(values(hits))
    if max_hits == 0
        return umat  # No data mapped, return standard U-matrix
    end

    # Compute U* = U * (1 - normalized_density)
    ustar = Dict{Tuple{Int,Int}, T}()
    for (pos, u_val) in umat
        norm_density = T(hits[pos]) / T(max_hits)
        ustar[pos] = u_val * (one(T) - norm_density)
    end

    return ustar
end

# ============================================================================
# Component Planes
# ============================================================================

"""
    compute_component_plane(som, dim::Int) -> Dict{Tuple{Int,Int}, T}

Extract the weight values for a single input dimension across all neurons.

Component planes show how each input feature is distributed across the map.

# Arguments
- `som`: Trained DBGSOM
- `dim::Int`: Input dimension index (1 to input_dim)

# Returns
Dict mapping position => weight value for dimension `dim`
"""
function compute_component_plane(som::DBGSOM{T}, dim::Int) where T
    @assert 1 <= dim <= input_dim(som) "dim must be in [1, $(input_dim(som))]"

    plane = Dict{Tuple{Int,Int}, T}()

    for (pos, neuron) in som.topology.neurons
        plane[pos] = neuron.weights[dim]
    end

    return plane
end

"""
    compute_all_component_planes(som) -> Vector{Dict{Tuple{Int,Int}, T}}

Extract component planes for all input dimensions.

# Returns
Vector of component plane dicts, one per input dimension
"""
function compute_all_component_planes(som::DBGSOM{T}) where T
    return [compute_component_plane(som, d) for d in 1:input_dim(som)]
end

# ============================================================================
# Hit Counts
# ============================================================================

"""
    compute_hit_counts(som, X) -> Dict{Tuple{Int,Int}, Int}

Count how many samples map to each neuron.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data (input_dim × n_samples)

# Returns
Dict mapping position => number of samples assigned to that neuron
"""
function compute_hit_counts(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    bmus = find_bmus(som, X)

    counts = Dict{Tuple{Int,Int}, Int}()

    # Initialize all positions to 0
    for pos in positions(som)
        counts[pos] = 0
    end

    # Count BMU assignments
    for bmu in bmus
        counts[bmu] += 1
    end

    return counts
end

# ============================================================================
# Topology Edges
# ============================================================================

"""
    compute_topology_edges(som) -> Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}

Get all edges in the topology graph for visualization.

# Returns
Vector of (pos1, pos2) tuples representing edges between neighboring neurons
"""
function compute_topology_edges(som::DBGSOM)
    edges = Set{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}()

    for pos in positions(som)
        for neighbor_pos in get_neighbors(pos)
            if has_neuron(som, neighbor_pos)
                # Use canonical ordering to avoid duplicates
                edge = pos < neighbor_pos ? (pos, neighbor_pos) : (neighbor_pos, pos)
                push!(edges, edge)
            end
        end
    end

    return collect(edges)
end

# ============================================================================
# Combined Plotting Data
# ============================================================================

"""
    get_plotting_data(som, X) -> PlottingData{T}

Extract all visualization data from a trained DBGSOM.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data for computing hit counts

# Returns
PlottingData struct containing positions, weights, U-matrix, hit counts, edges, and bounds

# Example
```julia
data = get_plotting_data(som, X)

# Use with your favorite plotting library:
# scatter(data.positions, color=values(data.umatrix))
```
"""
function get_plotting_data(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    pos_list = collect(positions(som))
    _, weights = get_positions_and_weights(som)

    PlottingData{T}(
        pos_list,
        weights,
        compute_umatrix(som),
        compute_hit_counts(som, X),
        compute_topology_edges(som),
        grid_bounds(som)
    )
end

# ============================================================================
# Data Conversion Utilities
# ============================================================================

"""
    to_matrix(data::Dict{Tuple{Int,Int}, T}, bounds) -> Matrix{Union{T, Missing}}

Convert position-keyed dict to a dense matrix for heatmap plotting.

Missing values are used for empty grid positions.

# Arguments
- `data`: Dict mapping (x, y) => value
- `bounds`: ((x_min, x_max), (y_min, y_max))

# Returns
Matrix where M[y - y_min + 1, x - x_min + 1] = data[(x, y)]
"""
function to_matrix(data::Dict{Tuple{Int,Int}, T}, bounds::Tuple{Tuple{Int,Int}, Tuple{Int,Int}}) where T
    ((x_min, x_max), (y_min, y_max)) = bounds

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    M = Matrix{Union{T, Missing}}(missing, height, width)

    for ((x, y), val) in data
        M[y - y_min + 1, x - x_min + 1] = val
    end

    return M
end

"""
    to_coordinate_arrays(positions) -> (xs, ys)

Convert position tuples to separate x and y arrays for scatter plots.
"""
function to_coordinate_arrays(positions::Vector{Tuple{Int,Int}})
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    return xs, ys
end

"""
    to_edge_segments(edges) -> Vector{Tuple{Vector{Int}, Vector{Int}}}

Convert edges to line segment format for plotting.

# Returns
Vector of (xs, ys) tuples, each representing a line segment
"""
function to_edge_segments(edges::Vector{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}})
    segments = Tuple{Vector{Int}, Vector{Int}}[]

    for (p1, p2) in edges
        push!(segments, ([p1[1], p2[1]], [p1[2], p2[2]]))
    end

    return segments
end

# ============================================================================
# ASCII Visualization (No Dependencies)
# ============================================================================

"""
    format_grid_ascii(som; show_hits=false) -> String

Generate ASCII representation of the SOM grid.

# Arguments
- `som`: Trained DBGSOM
- `show_hits`: If true, show hit counts; otherwise show neuron presence

# Example Output
```
Grid (5x4):
  1 2 3 4 5
1 ● ● ● ● ·
2 ● ● ● ● ●
3 · ● ● ● ●
4 · · ● ● ·
```
"""
function format_grid_ascii(som::DBGSOM; show_hits::Bool=false)
    ((x_min, x_max), (y_min, y_max)) = grid_bounds(som)

    lines = String[]
    push!(lines, "Grid ($(x_max - x_min + 1)x$(y_max - y_min + 1)):")

    # Header row
    header = "  " * join([lpad(string(x), 2) for x in x_min:x_max], "")
    push!(lines, header)

    # Grid rows
    for y in y_min:y_max
        row = lpad(string(y), 2)
        for x in x_min:x_max
            if has_neuron(som, (x, y))
                if show_hits
                    neuron = get_neuron(som, (x, y))
                    row *= lpad(neuron.hits > 9 ? "+" : string(neuron.hits), 2)
                else
                    row *= " ●"
                end
            else
                row *= " ·"
            end
        end
        push!(lines, row)
    end

    return join(lines, "\n")
end

"""
    format_umatrix_ascii(som; levels=5) -> String

Generate ASCII heatmap of U-matrix.

# Arguments
- `som`: Trained DBGSOM
- `levels`: Number of intensity levels (default: 5)

# Example Output
```
U-Matrix:
  1 2 3 4 5
1 ░ ░ ▒ ░ ·
2 ░ ▒ ▓ ▒ ░
3 · ▒ █ ▓ ▒
4 · · ▒ ▒ ·
```
"""
function format_umatrix_ascii(som::DBGSOM; levels::Int=5)
    umat = compute_umatrix(som)

    if isempty(umat)
        return "Empty U-matrix"
    end

    ((x_min, x_max), (y_min, y_max)) = grid_bounds(som)

    # Normalize to [0, 1]
    vals = collect(values(umat))
    min_val, max_val = extrema(vals)
    range_val = max_val - min_val

    # ASCII gradient characters
    chars = [' ', '░', '▒', '▓', '█']

    lines = String[]
    push!(lines, "U-Matrix:")

    header = "  " * join([lpad(string(x), 2) for x in x_min:x_max], "")
    push!(lines, header)

    for y in y_min:y_max
        row = lpad(string(y), 2)
        for x in x_min:x_max
            if haskey(umat, (x, y))
                val = umat[(x, y)]
                normalized = range_val > 0 ? (val - min_val) / range_val : 0.0
                level = clamp(round(Int, normalized * (levels - 1)) + 1, 1, levels)
                row *= " " * string(chars[level])
            else
                row *= " ·"
            end
        end
        push!(lines, row)
    end

    return join(lines, "\n")
end

# ============================================================================
# Neuron Clustering
# ============================================================================

"""
    cluster_neurons_umatrix(som, X; threshold_quantile=0.6) -> Dict{Tuple{Int,Int}, Int}

Cluster neurons using U* matrix (U-matrix combined with density) watershed approach.

The U* matrix formula: U*(i) = U(i) * (1 - normalized_density(i))
This combines topology (U-matrix) with data density (hit counts) for better clustering.

Finds connected regions of neurons where U* values are below the threshold.
This respects SOM topology and finds natural cluster boundaries.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data (input_dim × n_samples) used to compute density
- `threshold_quantile`: Quantile for threshold (0.0-1.0). Higher = fewer clusters. Default 0.6.

# Returns
Dict mapping position => cluster label (1 to k, where k is auto-determined)
"""
function cluster_neurons_umatrix(som::DBGSOM{T}, X::AbstractMatrix{T};
                                  threshold_quantile::Real=0.6) where T
    umat = compute_ustar_matrix(som, X)

    if isempty(umat)
        return Dict{Tuple{Int,Int}, Int}()
    end

    # Use quantile of U* values as threshold (higher quantile = more neurons below = fewer clusters)
    ustar_values = collect(values(umat))
    threshold = quantile(ustar_values, threshold_quantile)

    # Find connected components of low U-matrix neurons
    labels = Dict{Tuple{Int,Int}, Int}()
    current_label = 0

    for pos in keys(umat)
        if haskey(labels, pos)
            continue  # Already labeled
        end

        if umat[pos] <= threshold
            # Start new cluster with flood fill
            current_label += 1
            _flood_fill!(labels, umat, pos, current_label, threshold)
        end
    end

    # Label remaining high U-matrix neurons by nearest low neighbor
    for pos in keys(umat)
        if !haskey(labels, pos)
            # Find nearest labeled neighbor
            best_label = 0
            min_dist = Inf
            for neighbor in get_neighbors(pos)
                if haskey(labels, neighbor)
                    dist = umat[pos]
                    if dist < min_dist
                        min_dist = dist
                        best_label = labels[neighbor]
                    end
                end
            end
            if best_label > 0
                labels[pos] = best_label
            else
                # Isolated high-U neuron, assign to new cluster
                current_label += 1
                labels[pos] = current_label
            end
        end
    end

    return labels
end

"""
    cluster_neurons_n(som, X, n_clusters) -> Dict{Tuple{Int,Int}, Int}

Cluster neurons into exactly n_clusters using U* matrix + hierarchical merging.

First uses U* matrix clustering, then merges closest clusters by weight centroid
distance until the target number is reached.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data (input_dim × n_samples) used to compute density
- `n_clusters`: Target number of clusters

# Returns
Dict mapping position => cluster label (1 to n_clusters)
"""
function cluster_neurons_n(som::DBGSOM{T}, X::AbstractMatrix{T}, n_clusters::Int) where T
    # Start with U* clustering at high quantile (few initial clusters)
    labels = cluster_neurons_umatrix(som, X; threshold_quantile=0.99)

    if isempty(labels)
        return labels
    end

    current_n = length(unique(values(labels)))

    # If already at or below target, return as-is
    if current_n <= n_clusters
        return labels
    end

    # Get neuron weights for distance computation
    weights = Dict{Tuple{Int,Int}, Vector{T}}()
    for (pos, neuron) in som.topology.neurons
        weights[pos] = neuron.weights
    end

    # Hierarchically merge until we reach target
    while current_n > n_clusters
        # Find two closest clusters by centroid distance
        cluster_ids = unique(values(labels))
        min_dist = Inf
        merge_from, merge_to = 0, 0

        # Compute cluster centroids
        centroids = Dict{Int, Vector{T}}()
        cluster_positions = Dict{Int, Vector{Tuple{Int,Int}}}()

        for cid in cluster_ids
            cluster_positions[cid] = [p for (p, l) in labels if l == cid]
            cluster_weights = [weights[p] for p in cluster_positions[cid] if haskey(weights, p)]
            if !isempty(cluster_weights)
                centroids[cid] = mean(cluster_weights)
            end
        end

        # Find closest pair
        cids = collect(keys(centroids))
        for i in 1:length(cids)
            for j in (i+1):length(cids)
                c1, c2 = cids[i], cids[j]
                dist = norm(centroids[c1] - centroids[c2])
                if dist < min_dist
                    min_dist = dist
                    merge_from, merge_to = c1, c2
                end
            end
        end

        # Merge: relabel merge_from -> merge_to
        if merge_from > 0 && merge_to > 0
            for pos in keys(labels)
                if labels[pos] == merge_from
                    labels[pos] = merge_to
                end
            end
        end

        current_n = length(unique(values(labels)))
    end

    # Renumber labels to be contiguous 1:n
    old_labels = sort(unique(values(labels)))
    label_map = Dict(old => new for (new, old) in enumerate(old_labels))
    for pos in keys(labels)
        labels[pos] = label_map[labels[pos]]
    end

    return labels
end

"""
    _flood_fill!(labels, umat, start, label, threshold)

Helper: flood fill from start position, labeling connected low-U neurons.
"""
function _flood_fill!(labels::Dict{Tuple{Int,Int}, Int},
                      umat::Dict{Tuple{Int,Int}, T},
                      start::Tuple{Int,Int},
                      label::Int,
                      threshold::T) where T
    stack = [start]

    while !isempty(stack)
        pos = pop!(stack)

        if haskey(labels, pos)
            continue
        end

        if !haskey(umat, pos) || umat[pos] > threshold
            continue
        end

        labels[pos] = label

        # Add neighbors to stack
        for neighbor in get_neighbors(pos)
            if haskey(umat, neighbor) && !haskey(labels, neighbor)
                push!(stack, neighbor)
            end
        end
    end
end

# ============================================================================
# Quality Metrics (Topographic Error)
# ============================================================================

"""
    find_two_bmus(som, X) -> (bmu1s, bmu2s, dist1s, dist2s)

Find first and second Best Matching Units for all samples.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data (input_dim × n_samples)

# Returns
- `bmu1s::Vector{Tuple{Int,Int}}`: First BMU positions
- `bmu2s::Vector{Tuple{Int,Int}}`: Second BMU positions
- `dist1s::Vector{T}`: Distances to first BMUs
- `dist2s::Vector{T}`: Distances to second BMUs
"""
function find_two_bmus(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    n_samples = size(X, 2)
    pos_list, W = get_positions_and_weights(som)
    n_neurons_count = length(pos_list)

    # Compute all pairwise distances
    D = pairwise_distances_squared_auto(Matrix(X), W)

    bmu1s = Vector{Tuple{Int,Int}}(undef, n_samples)
    bmu2s = Vector{Tuple{Int,Int}}(undef, n_samples)
    dist1s = Vector{T}(undef, n_samples)
    dist2s = Vector{T}(undef, n_samples)

    @inbounds for i in 1:n_samples
        # Find two smallest distances
        row = view(D, i, :)

        # Find first minimum
        min1_idx = 1
        min1_val = row[1]
        for j in 2:n_neurons_count
            if row[j] < min1_val
                min1_val = row[j]
                min1_idx = j
            end
        end

        # Find second minimum
        min2_idx = min1_idx == 1 ? 2 : 1
        min2_val = row[min2_idx]
        for j in 1:n_neurons_count
            if j != min1_idx && row[j] < min2_val
                min2_val = row[j]
                min2_idx = j
            end
        end

        bmu1s[i] = pos_list[min1_idx]
        bmu2s[i] = pos_list[min2_idx]
        dist1s[i] = sqrt(min1_val)
        dist2s[i] = sqrt(min2_val)
    end

    return bmu1s, bmu2s, dist1s, dist2s
end

"""
    geodesic_distance(som, pos1, pos2) -> Int

Compute geodesic (shortest path) distance between two neurons on the SOM grid.

Uses BFS to find the shortest path, handling non-rectangular grids with holes.

# Returns
- Distance as number of grid steps, or -1 if positions are disconnected
"""
function geodesic_distance(som::DBGSOM, pos1::Tuple{Int,Int}, pos2::Tuple{Int,Int})
    if pos1 == pos2
        return 0
    end

    # BFS to find shortest path
    queue = Tuple{Tuple{Int,Int}, Int}[(pos1, 0)]
    visited = Set{Tuple{Int,Int}}([pos1])
    head = 1

    while head <= length(queue)
        current, dist = queue[head]
        head += 1

        # Check hex-connected neighbors
        for neighbor in get_neighbors(current)

            if neighbor == pos2
                return dist + 1
            end

            if has_neuron(som, neighbor) && neighbor ∉ visited
                push!(visited, neighbor)
                push!(queue, (neighbor, dist + 1))
            end
        end
    end

    # Disconnected
    return -1
end

"""
    compute_topographic_error(som, X) -> T

Compute topographic error: proportion of samples where BMU1 and BMU2 are not adjacent.

A topographic error of 0 indicates perfect topology preservation.
A topographic error of 1 indicates worst topology preservation.

# Arguments
- `som`: Trained DBGSOM
- `X`: Input data (input_dim × n_samples)

# Returns
Topographic error in range [0, 1]
"""
function compute_topographic_error(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    bmu1s, bmu2s, _, _ = find_two_bmus(som, X)
    n_samples = length(bmu1s)

    # Precompute neighbor sets for all neuron positions (IntraSOM-style approach)
    neighbor_sets = Dict{Tuple{Int,Int}, Set{Tuple{Int,Int}}}()
    for pos in positions(som.topology)
        neighbor_sets[pos] = Set(get_neighbors(pos))
    end

    n_errors = 0
    @inbounds for i in 1:n_samples
        if bmu2s[i] ∉ neighbor_sets[bmu1s[i]]
            n_errors += 1
        end
    end

    return T(n_errors) / T(n_samples)
end

# ============================================================================
# Summary
# ============================================================================

"""
    format_summary(som, X) -> String

Generate a text summary of the trained SOM.
"""
function format_summary(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    lines = String[]

    push!(lines, "DBGSOM Summary")
    push!(lines, "=" ^ 40)
    push!(lines, "")
    push!(lines, "Configuration:")
    push!(lines, "  Input dimension: $(input_dim(som))")
    if som.gt_method isa SpreadingFactor
        push!(lines, "  Spreading factor: $(som.sf)")
    elseif som.gt_method isa StatisticalError
        push!(lines, "  Lambda: $(som.gt_method.lambda)")
    else
        push!(lines, "  Growth method: $(typeof(som.gt_method))")
    end
    push!(lines, "  Max neurons: $(som.max_neurons)")
    push!(lines, "  Iterations: $(som.n_iter)")
    push!(lines, "")
    push!(lines, "Topology:")
    push!(lines, "  Neurons: $(n_neurons(som))")
    push!(lines, "  Bounds: $(grid_bounds(som))")
    push!(lines, "  Edges: $(length(compute_topology_edges(som)))")
    push!(lines, "")

    if som.trained
        qe = compute_quantization_error(som, X)
        hits = compute_hit_counts(som, X)
        umat = compute_umatrix(som)

        push!(lines, "Quality Metrics:")
        push!(lines, "  Quantization error: $(round(qe, digits=4))")
        push!(lines, "  Mean U-matrix: $(round(mean(values(umat)), digits=4))")
        push!(lines, "  Active neurons: $(count(v -> v > 0, values(hits))) / $(n_neurons(som))")
    else
        push!(lines, "Status: Not trained")
    end

    return join(lines, "\n")
end

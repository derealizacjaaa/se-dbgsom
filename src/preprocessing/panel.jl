"""
Panel data preprocessing for DBGSOM.

Provides sliding window transformation to convert panel/longitudinal data
into a format suitable for SOM training while preserving temporal dynamics.
"""

# ============================================================================
# PanelData Type
# ============================================================================

"""
    PanelData{T}

Container for panel/longitudinal data with entity and time dimensions.

# Fields
- `data::Dict{Any, Matrix{T}}`: Entity ID => (features × time_periods) matrix
- `feature_names::Vector{String}`: Names of features (optional)
- `n_features::Int`: Number of features per observation
"""
struct PanelData{T<:AbstractFloat}
    data::Dict{Any, Matrix{T}}
    feature_names::Vector{String}
    n_features::Int
end

"""
    PanelData{T}(data; feature_names=String[])

Create PanelData from a dictionary of entity => matrix mappings.

# Arguments
- `data::Dict`: Entity ID => (features × time_periods) matrix
- `feature_names::Vector{String}`: Optional feature names

# Example
```julia
panel = PanelData{Float64}(Dict(
    "firm_A" => rand(5, 10),  # 5 features, 10 time periods
    "firm_B" => rand(5, 12),  # 5 features, 12 time periods
))
```
"""
function PanelData{T}(data::Dict; feature_names::Vector{String}=String[]) where T
    @assert !isempty(data) "Panel data cannot be empty"

    # Infer n_features from first entity
    first_entity = first(values(data))
    n_features = size(first_entity, 1)

    # Validate all entities have same number of features
    for (entity, matrix) in data
        @assert size(matrix, 1) == n_features "Entity $entity has $(size(matrix,1)) features, expected $n_features"
    end

    # Validate feature names if provided
    if !isempty(feature_names)
        @assert length(feature_names) == n_features "feature_names length ($(length(feature_names))) must match n_features ($n_features)"
    end

    PanelData{T}(Dict{Any, Matrix{T}}(data), feature_names, n_features)
end

# Convenience constructor
PanelData(data::Dict{K, Matrix{T}}; kwargs...) where {K, T} =
    PanelData{T}(data; kwargs...)

# ============================================================================
# SlidingWindowResult Type
# ============================================================================

"""
    SlidingWindowResult{T}

Result of sliding window transformation, preserving entity and time metadata.

# Fields
- `X::Matrix{T}`: Transformed data (window_features × n_windows)
- `entity_ids::Vector{Any}`: Entity ID for each window
- `time_indices::Vector{Int}`: End time index for each window
- `window_size::Int`: Size of sliding window used
- `n_features::Int`: Original number of features
- `feature_names::Vector{String}`: Original feature names
"""
struct SlidingWindowResult{T<:AbstractFloat}
    X::Matrix{T}
    entity_ids::Vector{Any}
    time_indices::Vector{Int}
    window_size::Int
    n_features::Int
    feature_names::Vector{String}
end

# ============================================================================
# Sliding Window Transformation
# ============================================================================

"""
    sliding_windows(panel, window_size; step=1, normalize=false)

Transform panel data using sliding windows.

Each window captures `window_size` consecutive time periods, flattened into
a single feature vector. This preserves temporal dynamics for SOM training.

# Arguments
- `panel::PanelData{T}`: Panel data to transform
- `window_size::Int`: Number of time periods per window
- `step::Int`: Step size between windows (default: 1)
- `normalize::Bool`: Z-score normalize each feature across time (default: false)

# Returns
`SlidingWindowResult{T}` containing:
- `X`: Matrix of shape (n_features × window_size) × n_windows
- `entity_ids`: Entity ID for each window
- `time_indices`: End time index for each window

# Example
```julia
panel = PanelData{Float64}(Dict(
    "A" => rand(5, 10),  # 5 features, 10 time periods
    "B" => rand(5, 8)
))

result = sliding_windows(panel, 3)  # window of 3 periods
# result.X is 15 × n_windows (5 features × 3 periods)

fit!(som, result.X)
```
"""
function sliding_windows(
    panel::PanelData{T},
    window_size::Int;
    step::Int=1,
    normalize::Bool=false
) where T

    @assert window_size > 0 "window_size must be positive"
    @assert step > 0 "step must be positive"

    windows = Vector{Vector{T}}()
    entity_ids = Vector{Any}()
    time_indices = Vector{Int}()

    for (entity, data) in panel.data
        n_times = size(data, 2)

        if n_times < window_size
            @warn "Entity $entity has only $n_times time periods, skipping (need $window_size)"
            continue
        end

        # Optionally normalize within entity
        if normalize
            data = _zscore_columns(data)
        end

        # Create sliding windows
        for t in window_size:step:n_times
            window_start = t - window_size + 1
            window_data = data[:, window_start:t]

            # Flatten: concatenate columns (time-ordered)
            window_vec = vec(window_data)

            push!(windows, window_vec)
            push!(entity_ids, entity)
            push!(time_indices, t)
        end
    end

    @assert !isempty(windows) "No valid windows created. Check window_size vs time periods."

    # Convert to matrix (features × samples)
    X = reduce(hcat, windows)

    SlidingWindowResult{T}(
        X,
        entity_ids,
        time_indices,
        window_size,
        panel.n_features,
        panel.feature_names
    )
end

"""
    _zscore_columns(X)

Z-score normalize each row (feature) across columns (time).
Handles NaN values by computing mean/std only over valid values.
NaN values remain NaN after normalization.
"""
function _zscore_columns(X::Matrix{T}) where T
    result = similar(X)

    for i in 1:size(X, 1)  # Each feature row
        row = X[i, :]
        valid_mask = .!isnan.(row)

        if any(valid_mask)
            valid_vals = row[valid_mask]
            μ = mean(valid_vals)
            σ = std(valid_vals)
            σ = σ == 0 ? one(T) : σ

            # Normalize all values (NaN - μ = NaN, so NaN stays NaN)
            result[i, :] = (row .- μ) ./ σ
        else
            # All NaN row - keep as-is
            result[i, :] = row
        end
    end

    return result
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    n_windows(result::SlidingWindowResult)

Number of windows in the result.
"""
n_windows(result::SlidingWindowResult) = size(result.X, 2)

"""
    window_dim(result::SlidingWindowResult)

Dimensionality of each window vector.
"""
window_dim(result::SlidingWindowResult) = size(result.X, 1)

"""
    get_entity_windows(result, entity_id)

Get all window indices for a specific entity.
"""
function get_entity_windows(result::SlidingWindowResult, entity_id)
    return findall(==(entity_id), result.entity_ids)
end

"""
    get_entity_trajectory(result, entity_id, bmus)

Get the sequence of BMU positions for an entity over time.

# Arguments
- `result::SlidingWindowResult`: Sliding window result
- `entity_id`: Entity to track
- `bmus::Vector{Tuple{Int,Int}}`: BMU positions from predict(som, result.X)

# Returns
Vector of (time_index, bmu_position) tuples, sorted by time.
"""
function get_entity_trajectory(result::SlidingWindowResult, entity_id, bmus)
    indices = get_entity_windows(result, entity_id)

    trajectory = [(result.time_indices[i], bmus[i]) for i in indices]
    sort!(trajectory, by=first)

    return trajectory
end

"""
    unique_entities(result::SlidingWindowResult)

Get list of unique entity IDs.
"""
unique_entities(result::SlidingWindowResult) = unique(result.entity_ids)

"""
    entity_counts(result::SlidingWindowResult)

Count windows per entity.
"""
function entity_counts(result::SlidingWindowResult)
    counts = Dict{Any, Int}()
    for id in result.entity_ids
        counts[id] = get(counts, id, 0) + 1
    end
    return counts
end

# ============================================================================
# Window Feature Names
# ============================================================================

"""
    get_window_feature_names(result::SlidingWindowResult)

Generate feature names for the windowed data.

Returns names like "feature_t-2", "feature_t-1", "feature_t0" for a window of 3.
"""
function get_window_feature_names(result::SlidingWindowResult)
    names = String[]
    ws = result.window_size

    base_names = if isempty(result.feature_names)
        ["f$i" for i in 1:result.n_features]
    else
        result.feature_names
    end

    for t_offset in 0:(ws-1)
        t_label = t_offset - ws + 1  # -2, -1, 0 for window_size=3
        suffix = t_label == 0 ? "_t0" : "_t$(t_label)"

        for fname in base_names
            push!(names, fname * suffix)
        end
    end

    return names
end

# ============================================================================
# From DataFrame (convenience)
# ============================================================================

"""
    panel_from_columns(data, entity_col, time_col, feature_cols; T=Float64)

Create PanelData from column-oriented data (e.g., from CSV).

# Arguments
- `data`: Any indexable data structure with columns (DataFrame-like)
- `entity_col::Symbol`: Column containing entity IDs
- `time_col::Symbol`: Column containing time indices
- `feature_cols::Vector{Symbol}`: Columns containing features

# Example
```julia
# With a DataFrame-like structure
panel = panel_from_columns(df, :firm_id, :year, [:revenue, :profit, :assets])
```
"""
function panel_from_columns(
    data,
    entity_col::Symbol,
    time_col::Symbol,
    feature_cols::Vector{Symbol};
    T::Type{<:AbstractFloat}=Float64
)
    entities = unique(data[!, entity_col])
    n_features = length(feature_cols)

    panel_dict = Dict{Any, Matrix{T}}()

    for entity in entities
        mask = data[!, entity_col] .== entity
        entity_data = data[mask, :]

        # Sort by time
        time_order = sortperm(entity_data[!, time_col])
        entity_data = entity_data[time_order, :]

        # Extract features as matrix (features × time)
        n_times = size(entity_data, 1)
        matrix = Matrix{T}(undef, n_features, n_times)

        for (i, col) in enumerate(feature_cols)
            matrix[i, :] = T.(entity_data[!, col])
        end

        panel_dict[entity] = matrix
    end

    feature_names = String.(feature_cols)
    PanelData{T}(panel_dict; feature_names=feature_names)
end

"""
    panel_from_matrix(X, entity_ids, time_indices; feature_names=String[])

Create PanelData from a matrix with entity and time vectors.

# Arguments
- `X::Matrix{T}`: Data matrix (n_samples × n_features) or (n_features × n_samples)
- `entity_ids::Vector`: Entity ID for each sample
- `time_indices::Vector`: Time index for each sample
- `feature_names::Vector{String}`: Optional feature names
- `samples_in_rows::Bool`: If true, X is n_samples × n_features (default: true)

# Example
```julia
X = rand(100, 5)  # 100 samples, 5 features
ids = repeat(["A", "B", "C", "D"], inner=25)
times = repeat(1:25, outer=4)

panel = panel_from_matrix(X, ids, times)
```
"""
function panel_from_matrix(
    X::Matrix{T},
    entity_ids::Vector,
    time_indices::Vector;
    feature_names::Vector{String}=String[],
    samples_in_rows::Bool=true
) where T

    if samples_in_rows
        X = permutedims(X)  # Convert to features × samples
    end

    _, n_samples = size(X)

    @assert length(entity_ids) == n_samples "entity_ids length must match n_samples"
    @assert length(time_indices) == n_samples "time_indices length must match n_samples"

    # Group by entity
    panel_dict = Dict{Any, Matrix{T}}()

    for entity in unique(entity_ids)
        mask = entity_ids .== entity
        entity_X = X[:, mask]
        entity_times = time_indices[mask]

        # Sort by time
        time_order = sortperm(entity_times)
        panel_dict[entity] = entity_X[:, time_order]
    end

    PanelData{T}(panel_dict; feature_names=feature_names)
end

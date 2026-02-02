"""
Main DBGSOM training algorithm.
"""

# ============================================================================
# Training
# ============================================================================

"""
    fit!(som, X) -> DBGSOM

Train the DBGSOM on data X.

Training has two phases:
1. **Coarse phase** (with growth): Large neighborhood, neurons can be added
2. **Fine phase** (no growth): Small neighborhood, refine positions

Algorithm follows Vasighi & Amini (2017):
- GT computed via configured method (SpreadingFactor or StatisticalError)
- Error distribution from non-boundary to boundary neurons
- σ decays smoothly within each phase

# Arguments
- `som::DBGSOM{T}`: SOM to train
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)

# Returns
The trained SOM (same instance, mutated)
"""
function fit!(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    @assert size(X, 1) == input_dim(som) "Data dimension ($(size(X,1))) must match input_dim ($(input_dim(som)))"

    # Compute growth threshold using configured method
    # Pass X for StatisticalError; SpreadingFactor will ignore it
    gt = compute_growth_threshold(som, X)

    # Compute missingness info for NaN-aware training
    miss_info = if has_nan(X)
        compute_missingness_info(X)
    else
        nothing
    end

    # Split into coarse and fine phases
    n_coarse = div(som.n_iter, 2)
    n_fine = som.n_iter - n_coarse

    # === Phase 1: Coarse training with growth ===
    _train_coarse!(som, X, gt, n_coarse, miss_info)

    # === Phase 2: Fine training (no growth) ===
    _train_fine!(som, X, n_fine, miss_info)

    # === Cleanup: Prune dead neurons ===
    _prune!(som, X, miss_info)

    # Store missingness info for downstream use (prediction, clustering)
    som.missingness_info = miss_info

    som.trained = true
    return som
end

"""
    _train_coarse!(som, X, gt, n_epochs)

Coarse training phase with growth.

σ is computed once at the start from initial grid size,
then decays linearly throughout the phase.
"""
function _train_coarse!(som::DBGSOM{T}, X::AbstractMatrix{T}, gt::T, n_epochs::Int,
                        miss_info::Union{Nothing, MissingnessInfo{T}}=nothing) where T
    if n_epochs <= 0
        return
    end

    # Compute σ range ONCE at start of phase
    σ_start, σ_end = compute_sigma_range(som)

    for epoch in 1:n_epochs
        # Linear decay of σ
        σ = decay_sigma(epoch, n_epochs, σ_start, σ_end)

        bmus = dispatch_find_bmus(som.topology, X, miss_info)

        # Update weights
        if miss_info !== nothing
            update_weights_batch_nan_impute!(som.topology, X, bmus, σ, miss_info)
        else
            update_weights_batch!(som, X, bmus, σ)
        end

        # Accumulate errors
        reset_errors!(som)
        accumulate_errors!(som, X, bmus, miss_info)

        # Distribute errors from non-boundary to boundary neurons
        distribute_errors!(som, gt)

        # Grow from boundary neurons with high error
        grow!(som, gt)
    end
end

"""
    _train_fine!(som, X, n_epochs)

Fine training phase without growth.

σ continues to decay to allow fine-tuning of weights.
"""
function _train_fine!(som::DBGSOM{T}, X::AbstractMatrix{T}, n_epochs::Int,
                      miss_info::Union{Nothing, MissingnessInfo{T}}=nothing) where T
    if n_epochs <= 0
        return
    end

    # Compute σ range ONCE at start of fine phase
    σ_fine_start = compute_sigma(som) * T(0.3)
    σ_fine_end = T(0.5)

    for epoch in 1:n_epochs
        σ = decay_sigma(epoch, n_epochs, σ_fine_start, σ_fine_end)

        bmus = dispatch_find_bmus(som.topology, X, miss_info)

        # Update weights
        if miss_info !== nothing
            update_weights_batch_nan_impute!(som.topology, X, bmus, σ, miss_info)
        else
            update_weights_batch!(som, X, bmus, σ)
        end
    end
end

"""
    _prune!(som, X)

Final pruning step to remove dead neurons.
"""
function _prune!(som::DBGSOM{T}, X::AbstractMatrix{T},
                 miss_info::Union{Nothing, MissingnessInfo{T}}=nothing) where T
    # Update hit counts
    reset_errors!(som)
    bmus = dispatch_find_bmus(som.topology, X, miss_info)
    accumulate_errors!(som, X, bmus, miss_info)

    # Remove dead neurons
    prune_dead_neurons!(som)
end

# ============================================================================
# Prediction
# ============================================================================

"""
    predict(som, X) -> Vector{Tuple{Int,Int}}

Return BMU positions for each sample.

# Arguments
- `som::DBGSOM{T}`: Trained SOM
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)

# Returns
Vector of BMU grid positions
"""
function predict(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    @assert som.trained "Model must be trained before prediction"
    return dispatch_find_bmus(som.topology, X, som.missingness_info)
end

"""
    transform(som, X) -> Matrix{Int}

Return BMU coordinates as 2 × n_samples matrix.

# Arguments
- `som::DBGSOM{T}`: Trained SOM
- `X::AbstractMatrix{T}`: Input data (input_dim × n_samples)

# Returns
2 × n_samples matrix where row 1 is x-coordinates, row 2 is y-coordinates
"""
function transform(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    @assert som.trained "Model must be trained before transform"

    bmus = dispatch_find_bmus(som.topology, X, som.missingness_info)
    coords = Matrix{Int}(undef, 2, length(bmus))

    for (i, pos) in enumerate(bmus)
        coords[1, i] = pos[1]
        coords[2, i] = pos[2]
    end

    return coords
end

"""
    predict_with_distances(som, X) -> (Vector{Tuple{Int,Int}}, Vector{T})

Return BMU positions and distances for each sample.
"""
function predict_with_distances(som::DBGSOM{T}, X::AbstractMatrix{T}) where T
    @assert som.trained "Model must be trained before prediction"
    miss_info = som.missingness_info
    if miss_info !== nothing
        return find_bmus_with_distances(som.topology, X, miss_info)
    else
        return find_bmus_with_distances(som, X)
    end
end

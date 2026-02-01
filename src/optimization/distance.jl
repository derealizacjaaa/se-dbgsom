"""
Distance calculations for BasicDBGSOM.
"""

# ============================================================================
# Euclidean Distance
# ============================================================================

"""
    euclidean_squared(x, y) -> T

Squared Euclidean distance between two vectors.
"""
function euclidean_squared(x::AbstractVector{T}, y::AbstractVector{T}) where T
    dist = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        diff = x[i] - y[i]
        dist += diff * diff
    end
    return dist
end

"""
    euclidean(x, y) -> T

Euclidean distance between two vectors.
"""
euclidean(x::AbstractVector{T}, y::AbstractVector{T}) where T = sqrt(euclidean_squared(x, y))

# ============================================================================
# Pairwise Distances
# ============================================================================

"""
    pairwise_distances_squared!(D, X, W)

Compute squared distances between samples X and weight vectors W.
D[i,j] = ||X[:,i] - W[:,j]||²

# Arguments
- `D::Matrix{T}`: Output matrix (n_samples × n_neurons)
- `X::Matrix{T}`: Input data (input_dim × n_samples)
- `W::Matrix{T}`: Weight vectors (input_dim × n_neurons)
"""
function pairwise_distances_squared!(D::Matrix{T}, X::Matrix{T}, W::Matrix{T}) where T
    n_samples = size(X, 2)
    n_neurons = size(W, 2)

    @assert size(D) == (n_samples, n_neurons) "D must be n_samples × n_neurons"
    @assert size(X, 1) == size(W, 1) "X and W must have same input_dim"

    # D[i,j] = ||x_i||² + ||w_j||² - 2 * x_i' * w_j
    # Using efficient BLAS operations

    # Compute ||x_i||² for all samples
    x_norms = vec(sum(X .^ 2, dims=1))

    # Compute ||w_j||² for all neurons
    w_norms = vec(sum(W .^ 2, dims=1))

    # D = -2 * X' * W
    mul!(D, X', W, -2, 0)

    # Add norms
    @inbounds for j in 1:n_neurons
        for i in 1:n_samples
            D[i, j] += x_norms[i] + w_norms[j]
        end
    end
end

"""
    pairwise_distances_squared(X, W) -> Matrix{T}

Compute squared distances between samples X and weight vectors W.
"""
function pairwise_distances_squared(X::Matrix{T}, W::Matrix{T}) where T
    D = Matrix{T}(undef, size(X, 2), size(W, 2))
    pairwise_distances_squared!(D, X, W)
    return D
end

# ============================================================================
# NaN-Aware Distance Functions
# ============================================================================

"""
    euclidean_squared_nan(x, y) -> T

Squared Euclidean distance between two vectors, handling NaN values.

Only computes distance over dimensions where both vectors have valid (non-NaN) values.
The result is scaled by `total_dims / valid_dims` to make distances comparable
regardless of the amount of missing data.

Returns `typemax(T)` if no valid dimensions exist.
"""
function euclidean_squared_nan(x::AbstractVector{T}, y::AbstractVector{T}) where T
    dist = zero(T)
    valid_count = 0
    total_dim = length(x)

    @inbounds for i in eachindex(x, y)
        xi, yi = x[i], y[i]
        if !isnan(xi) && !isnan(yi)
            diff = xi - yi
            dist += diff * diff
            valid_count += 1
        end
    end

    # Scale to make comparable with full-dimension distances
    if valid_count > 0
        dist *= T(total_dim) / T(valid_count)
    else
        dist = typemax(T)  # No valid dimensions = infinite distance
    end

    return dist
end

"""
    euclidean_nan(x, y) -> T

Euclidean distance between two vectors, handling NaN values.
"""
euclidean_nan(x::AbstractVector{T}, y::AbstractVector{T}) where T = sqrt(euclidean_squared_nan(x, y))

"""
    pairwise_distances_squared_nan(X, W) -> Matrix{T}

Compute squared distances between samples X and weight vectors W, handling NaN values.

This version cannot use BLAS optimizations due to NaN handling requirements.
"""
function pairwise_distances_squared_nan(X::Matrix{T}, W::Matrix{T}) where T
    n_samples = size(X, 2)
    n_neurons = size(W, 2)
    D = Matrix{T}(undef, n_samples, n_neurons)

    @inbounds for j in 1:n_neurons
        w = view(W, :, j)
        for i in 1:n_samples
            x = view(X, :, i)
            D[i, j] = euclidean_squared_nan(x, w)
        end
    end

    return D
end

"""
    has_nan(X) -> Bool

Check if matrix contains any NaN values.
"""
function has_nan(X::AbstractMatrix{T}) where T
    @inbounds for i in eachindex(X)
        if isnan(X[i])
            return true
        end
    end
    return false
end

"""
    pairwise_distances_squared_auto(X, W) -> Matrix{T}

Compute squared distances, automatically choosing between BLAS-optimized version
(for complete data) and NaN-aware version (when missing data present).
"""
function pairwise_distances_squared_auto(X::Matrix{T}, W::Matrix{T}) where T
    if has_nan(X)
        return pairwise_distances_squared_nan(X, W)
    else
        return pairwise_distances_squared(X, W)
    end
end

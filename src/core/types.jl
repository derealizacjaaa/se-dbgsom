"""
Core data types for BasicDBGSOM.
"""

# ============================================================================
# Neuron
# ============================================================================

"""
    Neuron{T}

A single neuron in the SOM grid.

# Fields
- `weights::Vector{T}`: Weight vector (same dimension as input)
- `error::T`: Accumulated quantization error
- `hits::Int`: Number of times selected as BMU
"""
mutable struct Neuron{T<:AbstractFloat}
    weights::Vector{T}
    error::T
    hits::Int
end

Neuron{T}(weights::Vector{T}) where T = Neuron{T}(weights, zero(T), 0)

# ============================================================================
# Growth Threshold Methods
# ============================================================================

"""
    GrowthThresholdMethod

Abstract type for growth threshold computation strategies.

Two methods are available:
- `SpreadingFactor`: GT = -log(sf) × d (original DBGSOM)
- `StatisticalError`: GT = λ × sqrt(Σ std_i²) (data-adaptive)
"""
abstract type GrowthThresholdMethod end

"""
    SpreadingFactor{T}(sf)

Spreading factor method for growth threshold.

GT = -log(sf) × d where d is input dimensionality.

# Arguments
- `sf::T`: Spreading factor in (0, 1]
  - sf ≈ 0.1: Conservative growth (fewer neurons, high GT)
  - sf ≈ 0.5: Balanced growth (default)
  - sf ≈ 0.9: Aggressive growth (more neurons, low GT)
"""
struct SpreadingFactor{T<:AbstractFloat} <: GrowthThresholdMethod
    sf::T

    function SpreadingFactor{T}(sf::T) where T<:AbstractFloat
        @assert 0 < sf <= 1 "sf must be in (0, 1]"
        new{T}(sf)
    end
end

SpreadingFactor(sf::T) where T<:AbstractFloat = SpreadingFactor{T}(sf)
SpreadingFactor(sf::Real) = SpreadingFactor(Float64(sf))

"""
    StatisticalError{T}(lambda)

Statistical error method for growth threshold (data-adaptive).

GT = λ × sqrt(Σ std_i²) where std_i is the standard deviation
of dimension i across all training samples.

# Arguments
- `lambda::T`: Regulation coefficient (typically 0.5-2.0)
  - λ ≈ 0.5: Aggressive growth (lower threshold)
  - λ ≈ 1.0: Balanced growth (default)
  - λ ≈ 2.0: Conservative growth (higher threshold)

# Reference
Statistical approach for adaptive growth threshold in self-organizing maps.
"""
struct StatisticalError{T<:AbstractFloat} <: GrowthThresholdMethod
    lambda::T

    function StatisticalError{T}(lambda::T) where T<:AbstractFloat
        @assert lambda > 0 "lambda must be positive"
        new{T}(lambda)
    end
end

StatisticalError(lambda::T) where T<:AbstractFloat = StatisticalError{T}(lambda)
StatisticalError(lambda::Real) = StatisticalError(Float64(lambda))
StatisticalError{T}() where T = StatisticalError{T}(one(T))  # Default lambda=1.0

# ============================================================================
# SOMTopology
# ============================================================================

"""
    SOMTopology{T}

Grid topology storing neurons and their positions.

# Fields
- `neurons::Dict{Tuple{Int,Int}, Neuron{T}}`: Position => Neuron mapping
- `input_dim::Int`: Dimension of input vectors
"""
mutable struct SOMTopology{T<:AbstractFloat}
    neurons::Dict{Tuple{Int,Int}, Neuron{T}}
    input_dim::Int
end

"""
    SOMTopology{T}(; input_dim, init_size=(2,2))

Create a new topology with random initial weights.

Uses 0-based coordinates for compatibility with hexagonal odd-r offset coordinate system.
"""
function SOMTopology{T}(; input_dim::Int, init_size::Tuple{Int,Int}=(2, 2)) where T
    @assert input_dim > 0 "input_dim must be positive"
    @assert init_size[1] > 0 && init_size[2] > 0 "init_size must be positive"

    neurons = Dict{Tuple{Int,Int}, Neuron{T}}()

    # Initialize as hexagonal parallelogram with 0-based coordinates
    for y in 0:(init_size[2]-1)
        for x in 0:(init_size[1]-1)
            pos = (x, y)
            weights = randn(T, input_dim)
            neurons[pos] = Neuron{T}(weights)
        end
    end

    SOMTopology{T}(neurons, input_dim)
end

# ============================================================================
# DBGSOM
# ============================================================================

"""
    DBGSOM{T}

Directed Batch Growing Self-Organizing Map.

# Fields
- `topology::SOMTopology{T}`: Grid topology with neurons
- `gt_method::GrowthThresholdMethod`: Growth threshold computation method
- `max_neurons::Int`: Maximum neurons allowed
- `n_iter::Int`: Training iterations
- `trained::Bool`: Whether model has been trained

# Growth Threshold Methods
Two methods are available:
1. `SpreadingFactor(sf)`: GT = -log(sf) × d (original DBGSOM)
   - sf ≈ 0.1: Conservative, sf ≈ 0.9: Aggressive
2. `StatisticalError(λ)`: GT = λ × sqrt(Σ std_i²) (data-adaptive)
   - λ ≈ 0.5: Aggressive, λ ≈ 2.0: Conservative

# Examples
```julia
# Using spreading factor (original approach)
som = DBGSOM{Float64}(input_dim=10, sf=0.5)

# Using statistical error (data-adaptive)
som = DBGSOM{Float64}(input_dim=10, gt_method=StatisticalError(1.0))
```
"""
mutable struct DBGSOM{T<:AbstractFloat}
    topology::SOMTopology{T}
    gt_method::GrowthThresholdMethod
    max_neurons::Int
    n_iter::Int
    trained::Bool
end

"""
    DBGSOM{T}(; input_dim, sf=0.5, max_neurons=100, n_iter=100, init_size=(2,2))
    DBGSOM{T}(; input_dim, gt_method, max_neurons=100, n_iter=100, init_size=(2,2))

Create a new DBGSOM.

# Arguments
- `input_dim::Int`: Dimension of input data (required)
- `sf::Real`: Spreading factor in (0, 1] - uses SpreadingFactor method (default: 0.5)
- `gt_method::GrowthThresholdMethod`: Alternative: provide method directly
- `max_neurons::Int`: Maximum neurons (default: 100)
- `n_iter::Int`: Training iterations (default: 100)
- `init_size::Tuple{Int,Int}`: Initial grid size (default: (2,2))

# Examples
```julia
# Using spreading factor (original approach)
som = DBGSOM{Float64}(input_dim=10, sf=0.5)

# Using statistical error (data-adaptive)
som = DBGSOM{Float64}(input_dim=10, gt_method=StatisticalError(1.0))
```
"""
function DBGSOM{T}(;
    input_dim::Int,
    sf::Union{Real, Nothing}=nothing,
    gt_method::Union{GrowthThresholdMethod, Nothing}=nothing,
    max_neurons::Int=100,
    n_iter::Int=100,
    init_size::Tuple{Int,Int}=(2, 2)
) where T<:AbstractFloat

    # Determine growth threshold method
    if gt_method !== nothing && sf !== nothing
        @warn "Both sf and gt_method provided; using gt_method"
        method = gt_method
    elseif gt_method !== nothing
        method = gt_method
    elseif sf !== nothing
        @assert 0 < sf <= 1 "sf must be in (0, 1]"
        method = SpreadingFactor{T}(T(sf))
    else
        # Default: SpreadingFactor with sf=0.5
        method = SpreadingFactor{T}(T(0.5))
    end

    @assert max_neurons > 0 "max_neurons must be positive"
    @assert n_iter > 0 "n_iter must be positive"

    topology = SOMTopology{T}(input_dim=input_dim, init_size=init_size)

    DBGSOM{T}(topology, method, max_neurons, n_iter, false)
end

# Convenience constructor (Float64 default)
function DBGSOM(; input_dim::Int, sf::Union{Real,Nothing}=nothing,
                gt_method::Union{GrowthThresholdMethod,Nothing}=nothing, kwargs...)
    if gt_method !== nothing
        DBGSOM{Float64}(; input_dim=input_dim, gt_method=gt_method, kwargs...)
    elseif sf !== nothing
        DBGSOM{Float64}(; input_dim=input_dim, sf=sf, kwargs...)
    else
        DBGSOM{Float64}(; input_dim=input_dim, sf=0.5, kwargs...)
    end
end

# ============================================================================
# Accessors
# ============================================================================

n_neurons(som::DBGSOM) = length(som.topology.neurons)
n_neurons(topo::SOMTopology) = length(topo.neurons)

# Backward compatibility: allow som.sf access when using SpreadingFactor method
function Base.getproperty(som::DBGSOM, name::Symbol)
    if name === :sf
        method = getfield(som, :gt_method)
        if method isa SpreadingFactor
            return method.sf
        else
            error("DBGSOM using $(typeof(method)) does not have sf field. Access som.gt_method instead.")
        end
    else
        getfield(som, name)
    end
end

# Allow propertynames to show sf for SpreadingFactor method
function Base.propertynames(som::DBGSOM, private::Bool=false)
    base = (:topology, :gt_method, :max_neurons, :n_iter, :trained)
    if som.gt_method isa SpreadingFactor
        return (base..., :sf)
    else
        return base
    end
end

input_dim(som::DBGSOM) = som.topology.input_dim
input_dim(topo::SOMTopology) = topo.input_dim

positions(som::DBGSOM) = keys(som.topology.neurons)
positions(topo::SOMTopology) = keys(topo.neurons)

get_neuron(som::DBGSOM, pos::Tuple{Int,Int}) = som.topology.neurons[pos]
get_neuron(topo::SOMTopology, pos::Tuple{Int,Int}) = topo.neurons[pos]

has_neuron(som::DBGSOM, pos::Tuple{Int,Int}) = haskey(som.topology.neurons, pos)
has_neuron(topo::SOMTopology, pos::Tuple{Int,Int}) = haskey(topo.neurons, pos)

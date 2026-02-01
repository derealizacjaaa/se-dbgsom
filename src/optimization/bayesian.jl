"""
Bayesian Hyperparameter Optimization for DBGSOM.

Minimizes Quantization Error (QE) subject to Topographic Error (TE) constraint.

# Quick Start
```julia
using BasicDBGSOM

# Run optimization
result = bayesian_optimize(X;
    n_trials=30,
    te_constraint=0.25,
    verbose=true
)

# Use best model
best_som = result.best_model
println("Best QE: ", result.best_qe)
println("Best params: ", result.best_params)
```
"""

# ============================================================================
# Types
# ============================================================================

"""
    BayesianOptConfig{T}

Configuration for Bayesian optimization.

# Fields
- `gt_method_type`: :spreading_factor or :statistical_error
- `sf_range`: (min, max) for spreading factor (when using SpreadingFactor)
- `lambda_range`: (min, max) for lambda (when using StatisticalError)
- `max_neurons_range`: (min, max) for maximum neurons
- `n_iter_range`: (min, max) for training iterations
- `init_size_range`: (min, max) for initial grid size (both dimensions)
- `te_constraint`: Maximum allowed topographic error (default: 0.25)
- `n_trials`: Number of optimization trials
- `n_startup`: Number of random trials before using surrogate
- `seed`: Random seed for reproducibility
"""
struct BayesianOptConfig{T<:AbstractFloat}
    gt_method_type::Symbol  # :spreading_factor or :statistical_error
    sf_range::Tuple{T, T}
    lambda_range::Tuple{T, T}
    max_neurons_range::Tuple{Int, Int}
    n_iter_range::Tuple{Int, Int}
    init_size_range::Tuple{Int, Int}
    te_constraint::T
    n_trials::Int
    n_startup::Int
    seed::Union{Int, Nothing}
end

function BayesianOptConfig{T}(;
    gt_method_type::Symbol = :spreading_factor,
    sf_range::Tuple{<:Real, <:Real} = (0.1, 0.9),
    lambda_range::Tuple{<:Real, <:Real} = (0.3, 2.0),
    max_neurons_range::Tuple{Int, Int} = (20, 200),
    n_iter_range::Tuple{Int, Int} = (50, 200),
    init_size_range::Tuple{Int, Int} = (2, 5),
    te_constraint::Real = 0.25,
    n_trials::Int = 30,
    n_startup::Int = 10,
    seed::Union{Int, Nothing} = nothing
) where T
    @assert gt_method_type in (:spreading_factor, :statistical_error) "gt_method_type must be :spreading_factor or :statistical_error"
    BayesianOptConfig{T}(
        gt_method_type,
        (T(sf_range[1]), T(sf_range[2])),
        (T(lambda_range[1]), T(lambda_range[2])),
        max_neurons_range,
        n_iter_range,
        init_size_range,
        T(te_constraint),
        n_trials,
        n_startup,
        seed
    )
end

"""
    BayesianOptResult{T}

Result of Bayesian optimization.

# Fields
- `best_model`: Best trained DBGSOM model
- `best_params`: Best hyperparameters found
- `best_qe`: Best quantization error achieved
- `best_te`: Topographic error of best model
- `all_trials`: Vector of all trial results
- `config`: Configuration used
"""
struct BayesianOptResult{T<:AbstractFloat}
    best_model::Union{DBGSOM{T}, Nothing}
    best_params::Dict{Symbol, Any}
    best_qe::T
    best_te::T
    all_trials::Vector{Dict{Symbol, Any}}
    config::BayesianOptConfig{T}
end

"""
    TrialResult{T}

Result of a single optimization trial.
"""
struct TrialResult{T<:AbstractFloat}
    params::Dict{Symbol, Any}
    qe::T
    te::T
    feasible::Bool  # TE <= constraint
    score::T        # QE if feasible, else penalized
end

# ============================================================================
# Core Optimization
# ============================================================================

"""
    bayesian_optimize(X; kwargs...) -> BayesianOptResult

Find optimal DBGSOM hyperparameters using Bayesian optimization.

Minimizes Quantization Error (QE) subject to Topographic Error (TE) <= te_constraint.

# Arguments
- `X::AbstractMatrix{T}`: Training data (input_dim × n_samples)

# Keyword Arguments
- `gt_method_type`: :spreading_factor (default) or :statistical_error
- `sf_range`: (min, max) for spreading factor (default: (0.1, 0.9))
- `lambda_range`: (min, max) for lambda when using StatisticalError (default: (0.3, 2.0))
- `max_neurons_range`: (min, max) for max neurons (default: (20, 200))
- `n_iter_range`: (min, max) for iterations (default: (50, 200))
- `init_size_range`: (min, max) for initial grid size (default: (2, 5))
- `te_constraint`: Maximum TE allowed (default: 0.25)
- `n_trials`: Number of optimization trials (default: 30)
- `n_startup`: Random trials before surrogate (default: 10)
- `seed`: Random seed (default: nothing)
- `verbose`: Print progress (default: true)

# Returns
`BayesianOptResult` containing:
- `best_model`: Best trained DBGSOM
- `best_params`: Dict with :sf or :lambda, :max_neurons, :n_iter, :init_size
- `best_qe`: Best QE achieved
- `best_te`: TE of best model
- `all_trials`: All trial results

# Example
```julia
# Using SpreadingFactor (default)
result = bayesian_optimize(X;
    n_trials=50,
    te_constraint=0.25,
    sf_range=(0.2, 0.8)
)

# Using StatisticalError
result = bayesian_optimize(X;
    gt_method_type=:statistical_error,
    lambda_range=(0.5, 1.5),
    n_trials=50,
    te_constraint=0.25
)
```
"""
function bayesian_optimize(
    X::AbstractMatrix{T};
    gt_method_type::Symbol = :spreading_factor,
    sf_range::Tuple{<:Real, <:Real} = (0.1, 0.9),
    lambda_range::Tuple{<:Real, <:Real} = (0.3, 2.0),
    max_neurons_range::Tuple{Int, Int} = (20, 200),
    n_iter_range::Tuple{Int, Int} = (50, 200),
    init_size_range::Tuple{Int, Int} = (2, 5),
    te_constraint::Real = 0.25,
    n_trials::Int = 30,
    n_startup::Int = 10,
    seed::Union{Int, Nothing} = nothing,
    verbose::Bool = true
) where T<:AbstractFloat

    config = BayesianOptConfig{T}(;
        gt_method_type=gt_method_type,
        sf_range=sf_range,
        lambda_range=lambda_range,
        max_neurons_range=max_neurons_range,
        n_iter_range=n_iter_range,
        init_size_range=init_size_range,
        te_constraint=te_constraint,
        n_trials=n_trials,
        n_startup=n_startup,
        seed=seed
    )

    return _run_optimization(X, config, verbose)
end

"""
    _run_optimization(X, config, verbose) -> BayesianOptResult

Internal optimization loop using TPE-like sampling.
"""
function _run_optimization(
    X::AbstractMatrix{T},
    config::BayesianOptConfig{T},
    verbose::Bool
) where T

    # Set random seed
    if config.seed !== nothing
        Random.seed!(config.seed)
    end

    input_d = size(X, 1)
    n_samples = size(X, 2)

    if verbose
        println("=" ^ 60)
        println("  Bayesian Hyperparameter Optimization")
        println("=" ^ 60)
        method_name = config.gt_method_type == :spreading_factor ? "SpreadingFactor" : "StatisticalError"
        println("  Method: $method_name")
        println("  Objective: Minimize QE subject to TE ≤ $(config.te_constraint)")
        println("  Trials: $(config.n_trials) ($(config.n_startup) startup)")
        println("  Data: $input_d features × $n_samples samples")
        println("-" ^ 60)
        println("  Search ranges:")
        if config.gt_method_type == :spreading_factor
            println("    sf:          $(config.sf_range)")
        else
            println("    lambda:      $(config.lambda_range)")
        end
        println("    max_neurons: $(config.max_neurons_range)")
        println("    n_iter:      $(config.n_iter_range)")
        println("    init_size:   $(config.init_size_range)")
        println("=" ^ 60)
        println()
    end

    # Storage for trials
    trials = TrialResult{T}[]

    # Best feasible result tracking
    best_qe = T(Inf)
    best_te = T(Inf)
    best_params = Dict{Symbol, Any}()
    best_model::Union{DBGSOM{T}, Nothing} = nothing

    for trial_idx in 1:config.n_trials
        # Sample parameters
        if trial_idx <= config.n_startup
            # Random sampling for startup
            params = _sample_random(config)
        else
            # TPE-like sampling based on good/bad split
            params = _sample_tpe(config, trials)
        end

        # Evaluate
        result = _evaluate_trial(X, params, config)
        push!(trials, result)

        # Update best if feasible and better
        if result.feasible && result.qe < best_qe
            best_qe = result.qe
            best_te = result.te
            best_params = result.params

            # Retrain best model (we don't store during evaluation to save memory)
            best_model = _create_and_train(X, params)
        end

        # Progress output
        if verbose
            status = result.feasible ? "✓" : "✗"
            best_str = result.feasible && result.qe == best_qe ? " ★" : ""
            if config.gt_method_type == :spreading_factor
                @printf("  [%3d/%3d] %s QE=%.4f TE=%.3f | sf=%.3f neurons=%3d iter=%3d init=(%d,%d)%s\n",
                    trial_idx, config.n_trials, status,
                    result.qe, result.te,
                    params[:sf], params[:max_neurons], params[:n_iter],
                    params[:init_size][1], params[:init_size][2],
                    best_str)
            else
                @printf("  [%3d/%3d] %s QE=%.4f TE=%.3f | λ=%.3f neurons=%3d iter=%3d init=(%d,%d)%s\n",
                    trial_idx, config.n_trials, status,
                    result.qe, result.te,
                    params[:lambda], params[:max_neurons], params[:n_iter],
                    params[:init_size][1], params[:init_size][2],
                    best_str)
            end
        end
    end

    # Summary
    if verbose
        println()
        println("=" ^ 60)
        println("  Optimization Complete")
        println("=" ^ 60)

        n_feasible = count(t -> t.feasible, trials)
        println("  Feasible trials: $n_feasible / $(config.n_trials)")

        if best_model !== nothing
            println()
            println("  Best Parameters:")
            if config.gt_method_type == :spreading_factor
                println("    sf:          $(best_params[:sf])")
            else
                println("    lambda:      $(best_params[:lambda])")
            end
            println("    max_neurons: $(best_params[:max_neurons])")
            println("    n_iter:      $(best_params[:n_iter])")
            println("    init_size:   $(best_params[:init_size])")
            println()
            println("  Best Metrics:")
            println("    QE: $(round(best_qe, digits=4))")
            println("    TE: $(round(best_te, digits=4)) (constraint: ≤ $(config.te_constraint))")
            println("    Neurons: $(n_neurons(best_model))")
        else
            println()
            println("  ⚠ No feasible solution found!")
            println("  Consider relaxing te_constraint or expanding search ranges.")
        end
        println("=" ^ 60)
    end

    # Convert trials to dict format for result
    all_trials = [
        Dict{Symbol, Any}(
            :params => t.params,
            :qe => t.qe,
            :te => t.te,
            :feasible => t.feasible,
            :score => t.score
        )
        for t in trials
    ]

    return BayesianOptResult{T}(
        best_model,
        best_params,
        best_qe,
        best_te,
        all_trials,
        config
    )
end

# ============================================================================
# Sampling Strategies
# ============================================================================

"""
    _sample_random(config) -> Dict{Symbol, Any}

Sample parameters uniformly at random.
"""
function _sample_random(config::BayesianOptConfig{T}) where T
    params = Dict{Symbol, Any}(
        :max_neurons => rand(config.max_neurons_range[1]:config.max_neurons_range[2]),
        :n_iter => rand(config.n_iter_range[1]:config.n_iter_range[2]),
        :init_size => (
            rand(config.init_size_range[1]:config.init_size_range[2]),
            rand(config.init_size_range[1]:config.init_size_range[2])
        )
    )

    # Add method-specific parameter
    if config.gt_method_type == :spreading_factor
        params[:sf] = rand() * (config.sf_range[2] - config.sf_range[1]) + config.sf_range[1]
    else
        params[:lambda] = rand() * (config.lambda_range[2] - config.lambda_range[1]) + config.lambda_range[1]
    end

    return params
end

"""
    _sample_tpe(config, trials) -> Dict{Symbol, Any}

Sample using Tree-structured Parzen Estimator (TPE) approach.

Splits trials into good (top 25%) and bad groups, samples from good distribution.
"""
function _sample_tpe(config::BayesianOptConfig{T}, trials::Vector{TrialResult{T}}) where T
    # Sort trials by score (lower is better)
    sorted_trials = sort(trials, by=t -> t.score)

    # Split into good (top 25%) and bad
    n_good = max(1, div(length(sorted_trials), 4))
    good_trials = sorted_trials[1:n_good]

    # If no good feasible trials, fall back to random
    if isempty(good_trials)
        return _sample_random(config)
    end

    # Sample from good trials with noise (exploitation + exploration)
    good_params = [t.params for t in good_trials]

    # Pick a random good trial as base
    base = rand(good_params)

    # Add Gaussian noise for continuous params, resample discrete
    noise_scale = T(0.1)  # Exploration noise

    # For discrete params, either keep or resample nearby
    max_neurons = if rand() < 0.7
        # Perturb from good value
        base[:max_neurons] + rand(-20:20)
    else
        # Random
        rand(config.max_neurons_range[1]:config.max_neurons_range[2])
    end
    max_neurons = clamp(max_neurons, config.max_neurons_range[1], config.max_neurons_range[2])

    n_iter = if rand() < 0.7
        base[:n_iter] + rand(-20:20)
    else
        rand(config.n_iter_range[1]:config.n_iter_range[2])
    end
    n_iter = clamp(n_iter, config.n_iter_range[1], config.n_iter_range[2])

    init_x = if rand() < 0.7
        base[:init_size][1] + rand(-1:1)
    else
        rand(config.init_size_range[1]:config.init_size_range[2])
    end
    init_x = clamp(init_x, config.init_size_range[1], config.init_size_range[2])

    init_y = if rand() < 0.7
        base[:init_size][2] + rand(-1:1)
    else
        rand(config.init_size_range[1]:config.init_size_range[2])
    end
    init_y = clamp(init_y, config.init_size_range[1], config.init_size_range[2])

    params = Dict{Symbol, Any}(
        :max_neurons => max_neurons,
        :n_iter => n_iter,
        :init_size => (init_x, init_y)
    )

    # Method-specific parameter with noise
    if config.gt_method_type == :spreading_factor
        sf = base[:sf] + randn() * noise_scale * (config.sf_range[2] - config.sf_range[1])
        params[:sf] = clamp(sf, config.sf_range[1], config.sf_range[2])
    else
        lambda = base[:lambda] + randn() * noise_scale * (config.lambda_range[2] - config.lambda_range[1])
        params[:lambda] = clamp(lambda, config.lambda_range[1], config.lambda_range[2])
    end

    return params
end

# ============================================================================
# Evaluation
# ============================================================================

"""
    _evaluate_trial(X, params, config) -> TrialResult

Evaluate a single parameter configuration.
"""
function _evaluate_trial(
    X::AbstractMatrix{T},
    params::Dict{Symbol, Any},
    config::BayesianOptConfig{T}
) where T

    try
        # Create and train model
        som = _create_and_train(X, params)

        # Compute metrics
        qe = compute_quantization_error(som, X)
        te = compute_topographic_error(som, X)

        # Check feasibility
        feasible = te <= config.te_constraint

        # Score: QE if feasible, otherwise penalized
        score = if feasible
            qe
        else
            # Large penalty proportional to constraint violation
            qe + T(1000) * (te - config.te_constraint)
        end

        return TrialResult{T}(params, qe, te, feasible, score)

    catch e
        @warn "Bayesian optimization trial failed" exception=(e, catch_backtrace()) params=params
        return TrialResult{T}(params, T(Inf), T(1.0), false, T(Inf))
    end
end

"""
    _create_and_train(X, params) -> DBGSOM

Create and train a DBGSOM with given parameters.
Supports both SpreadingFactor (:sf) and StatisticalError (:lambda) methods.
"""
function _create_and_train(X::AbstractMatrix{T}, params::Dict{Symbol, Any}) where T
    input_d = size(X, 1)

    # Determine which method to use based on params
    if haskey(params, :sf)
        som = DBGSOM{T}(
            input_dim=input_d,
            sf=T(params[:sf]),
            max_neurons=params[:max_neurons],
            n_iter=params[:n_iter],
            init_size=params[:init_size]
        )
    else
        som = DBGSOM{T}(
            input_dim=input_d,
            gt_method=StatisticalError{T}(T(params[:lambda])),
            max_neurons=params[:max_neurons],
            n_iter=params[:n_iter],
            init_size=params[:init_size]
        )
    end

    fit!(som, X)
    return som
end

# ============================================================================
# Utilities
# ============================================================================

"""
    get_best_trials(result; n=10) -> Vector{Dict}

Get the n best feasible trials from optimization result.
"""
function get_best_trials(result::BayesianOptResult; n::Int=10)
    feasible = filter(t -> t[:feasible], result.all_trials)
    sorted = sort(feasible, by=t -> t[:qe])
    return sorted[1:min(n, length(sorted))]
end

"""
    summarize_trials(result) -> String

Generate a summary table of all trials.
"""
function summarize_trials(result::BayesianOptResult{T}) where T
    lines = String[]
    push!(lines, "=" ^ 70)
    push!(lines, "  Trial Summary (sorted by score)")
    push!(lines, "=" ^ 70)

    # Determine method type from first trial
    is_sf = !isempty(result.all_trials) && haskey(result.all_trials[1][:params], :sf)
    param_name = is_sf ? "SF" : "λ"

    push!(lines, @sprintf("%-6s %-8s %-8s %-8s %-6s %-6s %-6s %-8s",
        "Rank", "QE", "TE", "Status", param_name, "MaxN", "Iter", "Init"))
    push!(lines, "-" ^ 70)

    sorted = sort(result.all_trials, by=t -> t[:score])

    for (i, t) in enumerate(sorted[1:min(20, length(sorted))])
        status = t[:feasible] ? "✓" : "✗"
        param_val = is_sf ? t[:params][:sf] : t[:params][:lambda]
        push!(lines, @sprintf("%-6d %-8.4f %-8.4f %-8s %-6.3f %-6d %-6d (%d,%d)",
            i, t[:qe], t[:te], status,
            param_val, t[:params][:max_neurons], t[:params][:n_iter],
            t[:params][:init_size][1], t[:params][:init_size][2]))
    end

    if length(sorted) > 20
        push!(lines, "  ... and $(length(sorted) - 20) more trials")
    end
    push!(lines, "=" ^ 70)

    return join(lines, "\n")
end

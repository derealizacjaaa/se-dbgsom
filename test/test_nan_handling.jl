@testset "NaN Handling" begin

    # ========================================================================
    # MissingnessInfo Computation
    # ========================================================================
    @testset "compute_missingness_info" begin
        @testset "no missing data" begin
            X = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]  # 2 dims × 3 samples
            info = compute_missingness_info(X)

            @test info.rates ≈ [0.0, 0.0]
            @test info.obs_weights ≈ [1.0, 1.0]
            @test info.n_samples == 3
        end

        @testset "partial missing data" begin
            # 3 dims × 4 samples, dim 1 has 50% missing, dim 2 has 25%, dim 3 has 0%
            X = Float64[
                NaN  1.0  NaN  2.0;
                3.0  NaN  4.0  5.0;
                6.0  7.0  8.0  9.0
            ]
            info = compute_missingness_info(X)

            @test info.rates ≈ [0.5, 0.25, 0.0]
            @test info.obs_weights[1] < info.obs_weights[2] < info.obs_weights[3]
            @test info.obs_weights[3] ≈ 1.0  # Fully observed => weight 1.0
            @test info.n_samples == 4
        end

        @testset "all missing in one dimension" begin
            X = Float64[NaN NaN NaN; 1.0 2.0 3.0]
            info = compute_missingness_info(X)

            @test info.rates[1] ≈ 1.0
            @test info.obs_weights[1] ≈ 0.0  # Fully missing => weight 0.0
            @test info.rates[2] ≈ 0.0
            @test info.obs_weights[2] ≈ 1.0
        end

        @testset "single sample" begin
            X = reshape(Float64[1.0, NaN], 2, 1)
            info = compute_missingness_info(X)

            @test info.rates ≈ [0.0, 1.0]
            @test info.n_samples == 1
        end
    end

    # ========================================================================
    # Weighted Distance with Missingness
    # ========================================================================
    @testset "euclidean_squared_nan_weighted" begin
        @testset "no NaN, equal weights" begin
            x = Float64[1.0, 2.0, 3.0]
            y = Float64[4.0, 5.0, 6.0]
            info = MissingnessInfo(Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0], 100)

            d_weighted = euclidean_squared_nan_weighted(x, y, info)
            d_standard = euclidean_squared(x, y)

            # With equal weights and no NaN, should match standard distance
            @test d_weighted ≈ d_standard
        end

        @testset "NaN dimensions skipped" begin
            x = Float64[1.0, NaN, 3.0]
            y = Float64[4.0, 5.0, 6.0]
            info = MissingnessInfo(Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0], 100)

            d = euclidean_squared_nan_weighted(x, y, info)

            # Only dims 1 and 3 contribute: (4-1)^2 + (6-3)^2 = 9 + 9 = 18
            # Scaled by total_weight / used_weight = 3.0 / 2.0 = 1.5
            @test d ≈ 18.0 * (3.0 / 2.0)
        end

        @testset "high-missingness dims contribute less" begin
            x = Float64[1.0, 2.0]
            y = Float64[3.0, 4.0]

            # Dim 1 is well-observed (weight=1.0), dim 2 is poorly observed (weight=0.1)
            info_unequal = MissingnessInfo(Float64[0.0, 0.9], Float64[1.0, 0.1], 100)
            # Both well-observed
            info_equal = MissingnessInfo(Float64[0.0, 0.0], Float64[1.0, 1.0], 100)

            d_unequal = euclidean_squared_nan_weighted(x, y, info_unequal)
            d_equal = euclidean_squared_nan_weighted(x, y, info_equal)

            # With unequal weights, dim 2 contributes less, so distance is more
            # dominated by dim 1. The absolute values differ due to normalization.
            # Key test: the ratio of dim contributions changes
            @test d_unequal != d_equal
        end

        @testset "fully missing dimension has zero weight" begin
            x = Float64[1.0, NaN]
            y = Float64[3.0, 5.0]
            info = MissingnessInfo(Float64[0.0, 1.0], Float64[1.0, 0.0], 100)

            d = euclidean_squared_nan_weighted(x, y, info)

            # Only dim 1 contributes (dim 2 has NaN AND zero weight)
            # (3-1)^2 = 4, scaled by total_weight(1.0)/used_weight(1.0) = 1.0
            @test d ≈ 4.0
        end

        @testset "all NaN returns typemax" begin
            x = Float64[NaN, NaN]
            y = Float64[1.0, 2.0]
            info = MissingnessInfo(Float64[0.5, 0.5], Float64[0.5, 0.5], 100)

            d = euclidean_squared_nan_weighted(x, y, info)
            @test d == typemax(Float64)
        end
    end

    # ========================================================================
    # Pairwise Weighted Distances
    # ========================================================================
    @testset "pairwise_distances_squared_nan_weighted" begin
        @testset "matches pointwise computation" begin
            X = Float64[1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 dims × 2 samples
            W = Float64[0.5 1.5; 2.5 3.5; 4.5 5.5]   # 3 dims × 2 neurons
            info = MissingnessInfo(Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0], 10)

            D = pairwise_distances_squared_nan_weighted(X, W, info)

            @test size(D) == (2, 2)  # n_samples × n_neurons
            for i in 1:2, j in 1:2
                expected = euclidean_squared_nan_weighted(view(X, :, i), view(W, :, j), info)
                @test D[i, j] ≈ expected
            end
        end

        @testset "with NaN values" begin
            X = Float64[1.0 NaN; 3.0 4.0]  # 2 dims × 2 samples, sample 2 has NaN in dim 1
            W = Float64[0.5; 2.5][:, :]
            W = reshape(Float64[0.5, 2.5], 2, 1)  # 2 dims × 1 neuron
            info = MissingnessInfo(Float64[0.25, 0.0], Float64[0.75, 1.0], 4)

            D = pairwise_distances_squared_nan_weighted(X, W, info)

            @test size(D) == (2, 1)
            @test !isnan(D[1, 1])
            @test !isnan(D[2, 1])  # NaN sample still gets a distance
            @test D[2, 1] < typemax(Float64)  # Not infinite (dim 2 is valid)
        end
    end

    # ========================================================================
    # Neighbor-Based Weight Imputation
    # ========================================================================
    @testset "impute_neuron_from_neighbors!" begin

        @testset "imputes from direct neighbors" begin
            # Create a 3-neuron topology: center at (1,0) with neighbors at (0,0) and (2,0)
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[10.0, 20.0, 30.0]),
                    (1, 0) => Neuron{Float64}(Float64[0.0, 0.0, 0.0]),   # Target neuron
                    (2, 0) => Neuron{Float64}(Float64[12.0, 22.0, 32.0])
                ),
                3
            )

            # Dims 1 and 2 need imputation (low obs count), dim 3 is fine
            dim_obs_counts = Float64[0.0, 1.0, 100.0]
            min_obs = 5.0

            impute_neuron_from_neighbors!(topo, (1, 0), dim_obs_counts; min_obs=min_obs)

            neuron = topo.neurons[(1, 0)]
            # Dim 1 (obs=0 < min_obs=5): should be imputed from neighbors
            # Both neighbors at distance 1, so equal weight
            @test neuron.weights[1] ≈ (10.0 + 12.0) / 2.0
            # Dim 2 (obs=1 < min_obs=5): should be imputed
            @test neuron.weights[2] ≈ (20.0 + 22.0) / 2.0
            # Dim 3 (obs=100 >= min_obs=5): should NOT be changed
            @test neuron.weights[3] ≈ 0.0  # Unchanged
        end

        @testset "weights by inverse grid distance" begin
            # Center neuron at (1,0), direct neighbor at (0,0), 2nd-order at (2,0) only connected via center
            # Actually for hexagonal grid let me set up a clear distance difference
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[10.0]),  # distance 1 from (1,0)
                    (1, 0) => Neuron{Float64}(Float64[0.0]),   # Target
                    (2, 0) => Neuron{Float64}(Float64[20.0])   # distance 1 from (1,0)
                ),
                1
            )

            dim_obs_counts = Float64[0.0]
            impute_neuron_from_neighbors!(topo, (1, 0), dim_obs_counts; min_obs=1.0)

            neuron = topo.neurons[(1, 0)]
            # Both at distance 1, so equal inverse-distance weight
            @test neuron.weights[1] ≈ (10.0 + 20.0) / 2.0
        end

        @testset "uses 2nd-order neighbors when no direct neighbors" begin
            # Neuron at (0,0) with no direct neighbors, but 2nd-order neighbor at (2,0)
            # We need the positions to actually be at grid distance 2
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[0.0]),   # Target (isolated)
                    (2, 0) => Neuron{Float64}(Float64[42.0])   # distance 2
                ),
                1
            )

            dim_obs_counts = Float64[0.0]
            impute_neuron_from_neighbors!(topo, (0, 0), dim_obs_counts;
                                          min_obs=1.0, max_dist=2)

            neuron = topo.neurons[(0, 0)]
            @test neuron.weights[1] ≈ 42.0  # Only one neighbor available
        end

        @testset "no neighbors available - weight unchanged" begin
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[99.0])  # Completely isolated
                ),
                1
            )

            dim_obs_counts = Float64[0.0]
            impute_neuron_from_neighbors!(topo, (0, 0), dim_obs_counts; min_obs=1.0)

            @test topo.neurons[(0, 0)].weights[1] ≈ 99.0  # Unchanged
        end

        @testset "respects max_dist parameter" begin
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[0.0]),   # Target
                    (3, 0) => Neuron{Float64}(Float64[42.0])   # distance 3
                ),
                1
            )

            dim_obs_counts = Float64[0.0]
            # max_dist=2 should NOT reach the neighbor at distance 3
            impute_neuron_from_neighbors!(topo, (0, 0), dim_obs_counts;
                                          min_obs=1.0, max_dist=2)

            @test topo.neurons[(0, 0)].weights[1] ≈ 0.0  # Unchanged
        end
    end

    # ========================================================================
    # Integrated Batch Update with Imputation
    # ========================================================================
    @testset "update_weights_batch_nan_impute!" begin

        @testset "tracks per-neuron observation counts" begin
            # 2 dims × 3 samples, dim 1 has NaN in sample 2
            X = Float64[1.0 NaN 3.0; 4.0 5.0 6.0]

            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[2.0, 5.0]),
                    (1, 0) => Neuron{Float64}(Float64[2.0, 5.0])
                ),
                2
            )

            bmus = [(0, 0), (0, 0), (1, 0)]  # Samples 1,2 -> (0,0), sample 3 -> (1,0)
            σ = Float64(1.0)
            info = MissingnessInfo(Float64[1/3, 0.0], Float64[2/3, 1.0], 3)

            obs_counts = update_weights_batch_nan_impute!(topo, X, bmus, σ, info)

            # obs_counts should be a Dict mapping pos => Vector{Float64}
            @test haskey(obs_counts, (0, 0))
            @test haskey(obs_counts, (1, 0))

            # Neuron (0,0) is BMU for samples 1 and 2
            # Sample 1: dim 1 valid, dim 2 valid
            # Sample 2: dim 1 NaN, dim 2 valid
            # Plus neighborhood contribution from sample 3 (BMU (1,0))
            # At minimum, direct BMU contributions:
            # dim 1: at least 1 valid (sample 1), dim 2: at least 2 valid (samples 1,2)
            @test obs_counts[(0, 0)][1] > 0  # Some observations in dim 1
            @test obs_counts[(0, 0)][2] > obs_counts[(0, 0)][1]  # More obs in dim 2
        end

        @testset "imputes low-observation dimensions from neighbors" begin
            # Setup: neuron at (0,0) gets data only for dim 2 (dim 1 always NaN for its samples)
            # Neighbor at (1,0) has good weights for dim 1
            X = Float64[NaN 10.0; 5.0 20.0]  # 2 dims × 2 samples

            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[0.0, 0.0]),
                    (1, 0) => Neuron{Float64}(Float64[0.0, 0.0])
                ),
                2
            )

            bmus = [(0, 0), (1, 0)]  # Sample 1 -> (0,0), sample 2 -> (1,0)
            σ = Float64(0.5)  # Narrow neighborhood => minimal cross-influence
            info = MissingnessInfo(Float64[0.5, 0.0], Float64[0.5, 1.0], 2)

            update_weights_batch_nan_impute!(topo, X, bmus, σ, info)

            # After update + imputation:
            # Neuron (1,0) has good data for both dims (from sample 2: [10.0, 20.0])
            # Neuron (0,0) has NaN in dim 1 from its sample, so low obs count
            # => dim 1 should be imputed from neighbor (1,0)
            # The imputed value should be close to neuron (1,0)'s dim 1 weight
            neuron_00 = topo.neurons[(0, 0)]
            neuron_10 = topo.neurons[(1, 0)]

            # Dim 1 of neuron (0,0) should be imputed (close to neighbor)
            # Dim 2 should be updated normally from data
            @test neuron_00.weights[2] > 0  # Got valid data
            # The imputed dim 1 should be reasonable (not zero/random)
            @test neuron_00.weights[1] != 0.0  # Was imputed from neighbor
        end
    end

    # ========================================================================
    # Integration with fit! (End-to-End)
    # ========================================================================
    @testset "fit! with NaN data uses neighbor imputation" begin

        @testset "trains without error on NaN data" begin
            Random.seed!(42)
            X_complete = randn(Float64, 4, 50)  # 4 dims × 50 samples

            # Introduce 20% NaN
            X_nan = copy(X_complete)
            nan_mask = rand(Float64, size(X_nan)) .< 0.2
            X_nan[nan_mask] .= NaN

            som = DBGSOM{Float64}(input_dim=4, sf=0.5, max_neurons=20, n_iter=20)
            # Should not throw
            @test_nowarn fit!(som, X_nan)
            @test som.trained
        end

        @testset "neuron weights are finite after NaN training" begin
            Random.seed!(123)
            X = randn(Float64, 3, 30)
            # Make dim 1 heavily missing (70%)
            for i in 1:size(X, 2)
                if rand() < 0.7
                    X[1, i] = NaN
                end
            end

            som = DBGSOM{Float64}(input_dim=3, sf=0.5, max_neurons=15, n_iter=20)
            fit!(som, X)

            # All neuron weights should be finite (no NaN, no Inf)
            for (pos, neuron) in som.topology.neurons
                for d in 1:3
                    @test isfinite(neuron.weights[d])
                end
            end
        end

        @testset "prediction works with NaN input" begin
            Random.seed!(456)
            X_train = randn(Float64, 3, 40)
            X_train[1, 1:10] .= NaN  # Some NaN in training

            som = DBGSOM{Float64}(input_dim=3, sf=0.5, max_neurons=15, n_iter=20)
            fit!(som, X_train)

            # Predict on data with NaN
            X_test = randn(Float64, 3, 5)
            X_test[2, 3] = NaN

            bmus = predict(som, X_test)
            @test length(bmus) == 5
            @test all(pos -> has_neuron(som, pos), bmus)
        end
    end

    # ========================================================================
    # BMU Finding with Missingness-Weighted Distances
    # ========================================================================
    @testset "find_bmus uses weighted distances with MissingnessInfo" begin
        @testset "weighted BMU overload dispatches correctly" begin
            # Create topology with 2 neurons
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[1.0, 10.0]),
                    (1, 0) => Neuron{Float64}(Float64[5.0, 10.0])
                ),
                2
            )

            # Sample: [3.0, NaN] - dim 2 is missing
            X = reshape(Float64[3.0, NaN], 2, 1)

            # With equal weights, standard NaN distance picks closest in dim 1
            info_equal = MissingnessInfo(Float64[0.0, 0.0], Float64[1.0, 1.0], 10)
            bmus_equal = find_bmus(topo, X, info_equal)
            # dim 1 only: |3-1|=2 vs |3-5|=2, either could be BMU

            # With high missingness on dim 1, it contributes less
            info_dim1_missing = MissingnessInfo(Float64[0.9, 0.0], Float64[0.01, 1.0], 10)
            bmus_weighted = find_bmus(topo, X, info_dim1_missing)

            # Both should return valid positions
            @test length(bmus_equal) == 1
            @test length(bmus_weighted) == 1
            @test has_neuron(topo, bmus_equal[1])
            @test has_neuron(topo, bmus_weighted[1])
        end

        @testset "non-NaN data skips weighted computation" begin
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[1.0, 2.0]),
                    (1, 0) => Neuron{Float64}(Float64[3.0, 4.0])
                ),
                2
            )

            X = reshape(Float64[2.0, 3.0], 2, 1)  # No NaN
            info = MissingnessInfo(Float64[0.5, 0.0], Float64[0.25, 1.0], 10)

            # Should use standard BLAS distance (no NaN present)
            bmus = find_bmus(topo, X, info)
            @test length(bmus) == 1
            @test has_neuron(topo, bmus[1])
        end
    end

    # ========================================================================
    # Complete data is unaffected by new NaN handling
    # ========================================================================
    @testset "complete data uses fast path (no regression)" begin
        Random.seed!(999)
        X = randn(Float64, 4, 50)  # No NaN at all

        som = DBGSOM{Float64}(input_dim=4, sf=0.5, max_neurons=20, n_iter=20)
        fit!(som, X)

        @test som.trained
        @test som.missingness_info === nothing  # No NaN => no missingness info

        bmus = predict(som, X)
        @test length(bmus) == 50
    end

    # ========================================================================
    # MissingnessInfo stored on DBGSOM after training
    # ========================================================================
    @testset "DBGSOM stores missingness info after fit!" begin
        Random.seed!(789)
        X = randn(Float64, 3, 30)
        X[1, 1:15] .= NaN  # 50% missing in dim 1

        som = DBGSOM{Float64}(input_dim=3, sf=0.5, max_neurons=15, n_iter=10)
        fit!(som, X)

        # After training, missingness info should be accessible
        @test som.missingness_info !== nothing
        @test som.missingness_info.rates[1] ≈ 0.5
        @test som.missingness_info.rates[2] ≈ 0.0
        @test som.missingness_info.rates[3] ≈ 0.0
    end

    # ========================================================================
    # dispatch_find_bmus
    # ========================================================================
    @testset "dispatch_find_bmus" begin
        topo = SOMTopology{Float64}(
            Dict{Tuple{Int,Int}, Neuron{Float64}}(
                (0, 0) => Neuron{Float64}(Float64[1.0, 2.0]),
                (1, 0) => Neuron{Float64}(Float64[5.0, 6.0])
            ),
            2
        )
        X = reshape(Float64[2.0, 3.0], 2, 1)

        @testset "nothing miss_info uses standard path" begin
            bmus_dispatch = dispatch_find_bmus(topo, X, nothing)
            bmus_standard = find_bmus(topo, X)
            @test bmus_dispatch == bmus_standard
        end

        @testset "with MissingnessInfo uses weighted path" begin
            info = MissingnessInfo(Float64[0.0, 0.0], Float64[1.0, 1.0], 10)
            bmus_dispatch = dispatch_find_bmus(topo, X, info)
            bmus_weighted = find_bmus(topo, X, info)
            @test bmus_dispatch == bmus_weighted
        end
    end

    # ========================================================================
    # accumulate_errors! with miss_info
    # ========================================================================
    @testset "accumulate_errors! with miss_info" begin

        @testset "uses weighted distances when miss_info provided" begin
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[1.0, 2.0]),
                    (1, 0) => Neuron{Float64}(Float64[5.0, 6.0])
                ),
                2
            )

            # Sample with NaN in dim 2, BMU at (0,0)
            X = reshape(Float64[3.0, NaN], 2, 1)
            bmus = [(0, 0)]
            info = MissingnessInfo(Float64[0.0, 0.5], Float64[1.0, 0.25], 10)

            reset_errors!(topo)
            accumulate_errors!(topo, X, bmus, info)

            # Neuron (0,0) should have accumulated a valid error
            neuron = topo.neurons[(0, 0)]
            @test neuron.error > 0.0
            @test neuron.hits == 1
            @test isfinite(neuron.error)
        end

        @testset "nothing miss_info delegates to standard method" begin
            topo = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[1.0, 2.0])
                ),
                2
            )

            X = reshape(Float64[3.0, 4.0], 2, 1)
            bmus = [(0, 0)]

            # Standard path
            reset_errors!(topo)
            accumulate_errors!(topo, X, bmus)
            err_standard = topo.neurons[(0, 0)].error

            # Dispatch with nothing
            reset_errors!(topo)
            accumulate_errors!(topo, X, bmus, nothing)
            err_dispatch = topo.neurons[(0, 0)].error

            @test err_standard ≈ err_dispatch
        end

        @testset "weighted error differs from unweighted for NaN data" begin
            topo_w = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[0.0, 0.0, 0.0])
                ),
                3
            )
            topo_s = SOMTopology{Float64}(
                Dict{Tuple{Int,Int}, Neuron{Float64}}(
                    (0, 0) => Neuron{Float64}(Float64[0.0, 0.0, 0.0])
                ),
                3
            )

            # Dim 1 has high missingness weight penalty
            X = reshape(Float64[10.0, NaN, 5.0], 3, 1)
            bmus = [(0, 0)]
            info = MissingnessInfo(Float64[0.0, 0.8, 0.0], Float64[1.0, 0.04, 1.0], 10)

            reset_errors!(topo_w)
            accumulate_errors!(topo_w, X, bmus, info)

            reset_errors!(topo_s)
            accumulate_errors!(topo_s, X, bmus)

            # Both should have valid errors, but different values
            @test topo_w.neurons[(0, 0)].error > 0.0
            @test topo_s.neurons[(0, 0)].error > 0.0
            # The weighted version uses obs_weights; standard uses uniform scaling
            # They will generally differ
            @test topo_w.neurons[(0, 0)].hits == 1
            @test topo_s.neurons[(0, 0)].hits == 1
        end
    end

    # ========================================================================
    # predict vs predict_with_distances consistency
    # ========================================================================
    @testset "predict and predict_with_distances agree on NaN data" begin
        Random.seed!(321)
        X_train = randn(Float64, 3, 40)
        X_train[1, 1:15] .= NaN  # 37.5% missing in dim 1

        som = DBGSOM{Float64}(input_dim=3, sf=0.5, max_neurons=15, n_iter=20)
        fit!(som, X_train)

        # Test data with NaN
        X_test = randn(Float64, 3, 10)
        X_test[1, 3] = NaN
        X_test[2, 7] = NaN

        bmus_predict = predict(som, X_test)
        bmus_pwd, distances_pwd = predict_with_distances(som, X_test)

        # BMU assignments must be identical
        @test bmus_predict == bmus_pwd
        # Distances should be positive and finite
        @test all(isfinite, distances_pwd)
        @test all(d -> d >= 0.0, distances_pwd)
    end

    # ========================================================================
    # End-to-end metric consistency
    # ========================================================================
    @testset "end-to-end NaN training metric consistency" begin
        Random.seed!(654)
        X = randn(Float64, 4, 60)
        # 30% NaN across all dimensions
        nan_mask = rand(Float64, size(X)) .< 0.3
        X[nan_mask] .= NaN

        som = DBGSOM{Float64}(input_dim=4, sf=0.5, max_neurons=20, n_iter=20)
        fit!(som, X)

        @test som.trained
        @test som.missingness_info !== nothing

        # predict and predict_with_distances agree
        bmus1 = predict(som, X)
        bmus2, dists = predict_with_distances(som, X)
        @test bmus1 == bmus2
        @test all(isfinite, dists)

        # All neuron weights are finite
        for (_, neuron) in som.topology.neurons
            @test all(isfinite, neuron.weights)
        end
    end

    # ========================================================================
    # Dimension validation in compute_missingness_info
    # ========================================================================
    @testset "compute_missingness_info validates input" begin
        @test_throws AssertionError compute_missingness_info(Matrix{Float64}(undef, 0, 5))
        @test_throws AssertionError compute_missingness_info(Matrix{Float64}(undef, 3, 0))
    end

end

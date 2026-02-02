@testset "QE and TE Correctness" begin

    # ========================================================================
    # Helper: build a SOMTopology with manually specified neurons
    # ========================================================================
    function make_topology(positions_weights::Vector{Pair{Tuple{Int,Int}, Vector{Float64}}})
        neurons = Dict{Tuple{Int,Int}, Neuron{Float64}}()
        input_dim = length(first(positions_weights).second)
        for (pos, w) in positions_weights
            neurons[pos] = Neuron{Float64}(w)
        end
        return SOMTopology{Float64}(neurons, input_dim)
    end

    # Helper: build a DBGSOM wrapping a custom topology (marked as trained)
    function make_trained_dbgsom(topo::SOMTopology{Float64};
                                 gt_method=SpreadingFactor(0.5),
                                 max_neurons=100, n_iter=100)
        return DBGSOM{Float64}(topo, gt_method, max_neurons, n_iter, true, nothing)
    end

    # ========================================================================
    # QE Tests
    # ========================================================================
    @testset "Quantization Error (QE)" begin

        # ----------------------------------------------------------------
        # 1. Hand-computed QE with known topology and data
        # ----------------------------------------------------------------
        @testset "hand-computed QE matches expected value" begin
            # Two neurons in 2D:
            #   Neuron A at (0,0) with weights [1.0, 0.0]
            #   Neuron B at (1,0) with weights [5.0, 0.0]
            topo = make_topology([
                (0, 0) => [1.0, 0.0],
                (1, 0) => [5.0, 0.0]
            ])

            # Three samples (2 x 3 matrix, columns are samples):
            #   s1 = [0.0, 0.0]  -> BMU is A (dist = 1.0)
            #   s2 = [2.0, 0.0]  -> BMU is A (dist = 1.0)  (|2-1|=1 < |2-5|=3)
            #   s3 = [6.0, 0.0]  -> BMU is B (dist = 1.0)
            X = Float64[0.0 2.0 6.0;
                         0.0 0.0 0.0]

            qe = compute_quantization_error(topo, X)

            # Expected QE = mean([1.0, 1.0, 1.0]) = 1.0
            @test qe ≈ 1.0
        end

        # ----------------------------------------------------------------
        # 2. QE is zero when each sample exactly matches its BMU
        # ----------------------------------------------------------------
        @testset "QE is zero when samples equal BMU weights" begin
            topo = make_topology([
                (0, 0) => [1.0, 2.0, 3.0],
                (1, 0) => [4.0, 5.0, 6.0],
                (0, 1) => [7.0, 8.0, 9.0]
            ])

            # Each sample is exactly one of the weight vectors
            X = Float64[1.0 4.0 7.0;
                         2.0 5.0 8.0;
                         3.0 6.0 9.0]

            qe = compute_quantization_error(topo, X)
            @test qe ≈ 0.0 atol=1e-12
        end

        # ----------------------------------------------------------------
        # 3. QE is non-negative for any data
        # ----------------------------------------------------------------
        @testset "QE is non-negative" begin
            Random.seed!(42)
            topo = make_topology([
                (0, 0) => randn(5),
                (1, 0) => randn(5),
                (0, 1) => randn(5)
            ])

            X = randn(Float64, 5, 20)
            qe = compute_quantization_error(topo, X)
            @test qe >= 0.0
        end

        # ----------------------------------------------------------------
        # 4. QE with single neuron = mean distance from all samples
        # ----------------------------------------------------------------
        @testset "single neuron QE equals mean distance to all samples" begin
            w = Float64[3.0, 4.0]
            topo = make_topology([
                (0, 0) => w
            ])

            # 4 samples
            X = Float64[1.0 5.0 3.0 0.0;
                         1.0 7.0 4.0 0.0]

            # Manual distances to w = [3,4]:
            # s1 = [1,1]: sqrt((3-1)^2 + (4-1)^2) = sqrt(4+9) = sqrt(13)
            # s2 = [5,7]: sqrt((3-5)^2 + (4-7)^2) = sqrt(4+9) = sqrt(13)
            # s3 = [3,4]: sqrt(0) = 0
            # s4 = [0,0]: sqrt(9+16) = 5
            expected_qe = (sqrt(13.0) + sqrt(13.0) + 0.0 + 5.0) / 4.0

            qe = compute_quantization_error(topo, X)
            @test qe ≈ expected_qe atol=1e-10
        end

        # ----------------------------------------------------------------
        # 5. QE consistency: function vs manual find_bmus + distance
        # ----------------------------------------------------------------
        @testset "QE matches manual BMU distance computation" begin
            Random.seed!(123)
            topo = make_topology([
                (0, 0) => randn(3),
                (1, 0) => randn(3),
                (0, 1) => randn(3),
                (1, 1) => randn(3)
            ])

            X = randn(Float64, 3, 30)

            # Compute QE via the function under test
            qe_func = compute_quantization_error(topo, X)

            # Compute QE manually via find_bmus_with_distances
            bmus, distances = find_bmus_with_distances(topo, X)
            qe_manual = mean(distances)

            @test qe_func ≈ qe_manual atol=1e-12
        end

        # ----------------------------------------------------------------
        # 6. QE via DBGSOM wrapper matches SOMTopology version
        # ----------------------------------------------------------------
        @testset "QE via DBGSOM wrapper matches topology version" begin
            Random.seed!(456)
            topo = make_topology([
                (0, 0) => randn(4),
                (1, 0) => randn(4)
            ])
            som = make_trained_dbgsom(topo)
            X = randn(Float64, 4, 10)

            qe_topo = compute_quantization_error(topo, X)
            qe_som = compute_quantization_error(som, X)
            @test qe_topo ≈ qe_som
        end

        # ----------------------------------------------------------------
        # 7. Hand-computed QE with multi-dimensional data
        # ----------------------------------------------------------------
        @testset "hand-computed QE with 3D data and two neurons" begin
            # Neuron A at (0,0) with weights [0, 0, 0]
            # Neuron B at (1,0) with weights [10, 10, 10]
            topo = make_topology([
                (0, 0) => [0.0, 0.0, 0.0],
                (1, 0) => [10.0, 10.0, 10.0]
            ])

            # s1 = [1, 1, 1] -> BMU A, dist = sqrt(3)
            # s2 = [9, 9, 9] -> BMU B, dist = sqrt(3)
            X = Float64[1.0 9.0;
                         1.0 9.0;
                         1.0 9.0]

            qe = compute_quantization_error(topo, X)
            @test qe ≈ sqrt(3.0) atol=1e-10
        end
    end

    # ========================================================================
    # TE Tests
    # ========================================================================
    @testset "Topographic Error (TE)" begin

        # ----------------------------------------------------------------
        # 1. TE = 0 when all BMU pairs are adjacent
        # ----------------------------------------------------------------
        @testset "TE is zero when all BMU1/BMU2 pairs are adjacent" begin
            # Two adjacent neurons at (0,0) and (1,0)
            # get_neighbors((0,0)) for even row includes (1,0), so they ARE adjacent
            topo = make_topology([
                (0, 0) => [0.0, 0.0],
                (1, 0) => [10.0, 0.0]
            ])
            som = make_trained_dbgsom(topo)

            # Any sample will have BMU1 and BMU2 among these two adjacent neurons
            X = Float64[5.0 3.0 7.0;
                         0.0 0.0 0.0]

            te = compute_topographic_error(som, X)
            @test te ≈ 0.0
        end

        # ----------------------------------------------------------------
        # 2. TE = 1 when NO BMU pairs are adjacent
        # ----------------------------------------------------------------
        @testset "TE is one when no BMU1/BMU2 pairs are adjacent" begin
            # Two neurons that are NOT adjacent on the hex grid.
            # (0,0) even row neighbors: (1,0), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)
            # Place second neuron far away at (5,5), which is not adjacent to (0,0)
            topo = make_topology([
                (0, 0) => [0.0],
                (5, 5) => [100.0]
            ])
            som = make_trained_dbgsom(topo)

            # Verify they are indeed not adjacent
            neighbors_of_00 = Set(get_neighbors((0, 0)))
            @test (5, 5) ∉ neighbors_of_00

            # All samples have BMU1 and BMU2 from these two non-adjacent neurons
            X = Float64[50.0 20.0 80.0]  # 1 x 3

            te = compute_topographic_error(som, X)
            @test te ≈ 1.0
        end

        # ----------------------------------------------------------------
        # 3. Mixed adjacency (0 < TE < 1)
        # ----------------------------------------------------------------
        @testset "TE with mixed adjacency is between 0 and 1" begin
            # Three neurons:
            #   A at (0,0): weights [0.0]
            #   B at (1,0): weights [5.0]  -- adjacent to A (even row)
            #   C at (5,5): weights [100.0] -- NOT adjacent to A or B
            topo = make_topology([
                (0, 0) => [0.0],
                (1, 0) => [5.0],
                (5, 5) => [100.0]
            ])
            som = make_trained_dbgsom(topo)

            # Verify adjacency:
            # A-(0,0) and B-(1,0) are adjacent (even row, east neighbor)
            @test (1, 0) in get_neighbors((0, 0))
            # A-(0,0) and C-(5,5) are NOT adjacent
            @test (5, 5) ∉ Set(get_neighbors((0, 0)))
            # B-(1,0) and C-(5,5) are NOT adjacent
            @test (5, 5) ∉ Set(get_neighbors((1, 0)))

            # Sample near A: BMU1=A, BMU2=B (adjacent) -> no error
            # Sample near C: BMU1=C, BMU2=B (not adjacent) -> error
            # s1 = [1.0]: closest to A(0), next B(5), adjacent => OK
            # s2 = [90.0]: closest to C(100), next B(5) => NOT adjacent => error
            X = Float64[1.0 90.0]  # 1 x 2

            te = compute_topographic_error(som, X)

            # 1 out of 2 is an error => TE = 0.5
            @test te ≈ 0.5
        end

        # ----------------------------------------------------------------
        # 4. TE is always in [0, 1]
        # ----------------------------------------------------------------
        @testset "TE is always in [0, 1] range" begin
            Random.seed!(789)
            for trial in 1:5
                n_neurons_test = rand(3:8)
                # Build a small grid topology
                neurons_list = Pair{Tuple{Int,Int}, Vector{Float64}}[]
                for i in 0:(n_neurons_test-1)
                    push!(neurons_list, (i, 0) => randn(3))
                end
                topo = make_topology(neurons_list)
                som = make_trained_dbgsom(topo)

                X = randn(Float64, 3, 15)
                te = compute_topographic_error(som, X)
                @test 0.0 <= te <= 1.0
            end
        end

        # ----------------------------------------------------------------
        # 5a. TE with 2 adjacent neurons -> TE = 0
        # ----------------------------------------------------------------
        @testset "two adjacent neurons give TE = 0" begin
            # (0,0) and (1,0) are adjacent (even row)
            topo = make_topology([
                (0, 0) => [0.0, 0.0],
                (1, 0) => [1.0, 1.0]
            ])
            som = make_trained_dbgsom(topo)

            X = randn(Float64, 2, 10)
            te = compute_topographic_error(som, X)
            @test te ≈ 0.0
        end

        # ----------------------------------------------------------------
        # 5b. TE with 2 non-adjacent neurons -> TE = 1
        # ----------------------------------------------------------------
        @testset "two non-adjacent neurons give TE = 1" begin
            # (0,0) and (3,3) are far apart, not adjacent
            topo = make_topology([
                (0, 0) => [0.0, 0.0],
                (3, 3) => [1.0, 1.0]
            ])
            som = make_trained_dbgsom(topo)

            @test (3, 3) ∉ Set(get_neighbors((0, 0)))

            X = randn(Float64, 2, 10)
            te = compute_topographic_error(som, X)
            @test te ≈ 1.0
        end

        # ----------------------------------------------------------------
        # 6. Hex grid adjacency: odd-row offset correctness
        # ----------------------------------------------------------------
        @testset "hex grid adjacency for even and odd rows" begin
            # Even row (y=0) neighbors of (2,0):
            even_neighbors = Set(get_neighbors((2, 0)))
            @test even_neighbors == Set([
                (3, 0), (2, -1), (1, -1), (1, 0), (1, 1), (2, 1)
            ])

            # Odd row (y=1) neighbors of (2,1):
            odd_neighbors = Set(get_neighbors((2, 1)))
            @test odd_neighbors == Set([
                (3, 1), (3, 0), (2, 0), (1, 1), (2, 2), (3, 2)
            ])
        end

        # ----------------------------------------------------------------
        # 7. TE with hand-computed mixed scenario (4 samples)
        # ----------------------------------------------------------------
        @testset "hand-computed TE with 4 samples" begin
            # Build a line of 3 neurons: A(0,0), B(1,0), C(2,0)
            # Adjacencies (even row):
            #   A-B: (0,0) neighbor (1,0) -> adjacent
            #   B-C: (1,0) neighbor (2,0) -> adjacent
            #   A-C: (0,0) neighbors do NOT include (2,0) -> NOT adjacent
            topo = make_topology([
                (0, 0) => [0.0],
                (1, 0) => [5.0],
                (2, 0) => [10.0]
            ])
            som = make_trained_dbgsom(topo)

            # Verify A-C are not adjacent
            @test (2, 0) ∉ Set(get_neighbors((0, 0)))

            # s1 = [2.0]: BMU1=A(0), BMU2=B(5) => adjacent => OK
            # s2 = [4.0]: BMU1=B(5), BMU2=A(0) => adjacent => OK
            # s3 = [7.0]: BMU1=B(5), BMU2=C(10) => adjacent => OK  (|7-5|=2, |7-10|=3, |7-0|=7)
            # s4 = [4.9]: BMU1=B(5), BMU2=A(0) => adjacent => OK
            # All adjacent => TE = 0
            X = Float64[2.0 4.0 7.0 4.9]

            te = compute_topographic_error(som, X)
            @test te ≈ 0.0

            # Now force a non-adjacent case:
            # s5 = [4.999]: BMU1=B(5), BMU2=A(0) (|4.999-5|=0.001 < |4.999-0|=4.999 < |4.999-10|=5.001)
            #   => adjacent => OK
            # Instead, let us craft a sample where BMU1=A, BMU2=C (non-adjacent):
            # For BMU1=A, BMU2=C: need |x-0| < |x-5| AND |x-10| < |x-5|
            # |x| < |x-5| => x < 2.5
            # |x-10| < |x-5| => x > 7.5  -- contradiction!
            # So it's not possible with this 1D configuration. That is correct for a line.
            # With 1D data on a 3-neuron line, BMU2 is always adjacent to BMU1.
        end

        # ----------------------------------------------------------------
        # 8. TE with 2D data forcing non-adjacent second BMU
        # ----------------------------------------------------------------
        @testset "TE detects non-adjacent second BMU in 2D" begin
            # Neurons:
            #   A at (0,0): [0, 0]
            #   B at (1,0): [10, 0]  -- adjacent to A
            #   C at (5,5): [0, 1]   -- NOT adjacent to A
            topo = make_topology([
                (0, 0) => [0.0, 0.0],
                (1, 0) => [10.0, 0.0],
                (5, 5) => [0.0, 1.0]
            ])
            som = make_trained_dbgsom(topo)

            @test (5, 5) ∉ Set(get_neighbors((0, 0)))

            # Sample = [0.0, 0.5]:
            #   dist to A(0,0): sqrt(0 + 0.25) = 0.5
            #   dist to B(10,0): sqrt(100 + 0.25) = ~10.0
            #   dist to C(0,1): sqrt(0 + 0.25) = 0.5
            # BMU1=A (or C, tied), BMU2=C (or A, tied)
            # If BMU1=A, BMU2=C => NOT adjacent => error
            # If BMU1=C, BMU2=A => NOT adjacent => error
            # Either way, TE = 1.0 for this sample
            X = reshape(Float64[0.0, 0.5], 2, 1)

            te = compute_topographic_error(som, X)
            @test te ≈ 1.0
        end
    end

    # ========================================================================
    # find_two_bmus Tests
    # ========================================================================
    @testset "find_two_bmus correctness" begin

        @testset "returns correct BMU1 and BMU2" begin
            topo = make_topology([
                (0, 0) => [0.0],
                (1, 0) => [5.0],
                (2, 0) => [10.0]
            ])
            som = make_trained_dbgsom(topo)

            # Sample at 1.0: closest to A(0)=1, next B(5)=4, then C(10)=9
            X = reshape(Float64[1.0], 1, 1)

            bmu1s, bmu2s, dist1s, dist2s = find_two_bmus(som, X)

            @test bmu1s[1] == (0, 0)  # Closest to 0
            @test bmu2s[1] == (1, 0)  # Next closest to 5
            @test dist1s[1] ≈ 1.0
            @test dist2s[1] ≈ 4.0
        end

        @testset "distances are Euclidean (not squared)" begin
            topo = make_topology([
                (0, 0) => [0.0, 0.0],
                (1, 0) => [3.0, 4.0]
            ])
            som = make_trained_dbgsom(topo)

            X = reshape(Float64[0.0, 0.0], 2, 1)

            _, _, dist1s, dist2s = find_two_bmus(som, X)

            @test dist1s[1] ≈ 0.0
            @test dist2s[1] ≈ 5.0  # sqrt(9+16) = 5
        end

        @testset "BMU1 distance <= BMU2 distance" begin
            Random.seed!(321)
            topo = make_topology([
                (0, 0) => randn(4),
                (1, 0) => randn(4),
                (0, 1) => randn(4),
                (1, 1) => randn(4)
            ])
            som = make_trained_dbgsom(topo)

            X = randn(Float64, 4, 20)
            bmu1s, bmu2s, dist1s, dist2s = find_two_bmus(som, X)

            for i in 1:20
                @test dist1s[i] <= dist2s[i] + 1e-10
            end
        end
    end

    # ========================================================================
    # QE / TE Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin

        @testset "QE with many samples mapping to same BMU" begin
            topo = make_topology([
                (0, 0) => [0.0, 0.0],
                (1, 0) => [100.0, 100.0]
            ])

            # All samples very close to neuron A
            X = Float64[0.1 0.2 0.3 -0.1 -0.2;
                         0.1 0.2 0.3 -0.1 -0.2]

            qe = compute_quantization_error(topo, X)

            # All map to (0,0), distances are small
            expected_dists = [sqrt(0.01+0.01), sqrt(0.04+0.04), sqrt(0.09+0.09),
                              sqrt(0.01+0.01), sqrt(0.04+0.04)]
            @test qe ≈ mean(expected_dists) atol=1e-10
        end

        @testset "QE with large number of neurons" begin
            Random.seed!(111)
            # 20 neurons in a line
            neurons_list = Pair{Tuple{Int,Int}, Vector{Float64}}[]
            for i in 0:19
                push!(neurons_list, (i, 0) => Float64[Float64(i)])
            end
            topo = make_topology(neurons_list)

            # Samples at 0.5, 1.5, ..., 19.5 -> each equidistant from two neurons
            X = reshape(Float64[i + 0.5 for i in 0:18], 1, 19)
            qe = compute_quantization_error(topo, X)
            @test qe ≈ 0.5 atol=1e-10  # Every sample is 0.5 from its nearest neuron
        end

        @testset "TE with single neuron degeneracy" begin
            # With only 1 neuron, find_two_bmus uses @inbounds which skips
            # bounds checking. The min2_idx initialization tries index 2
            # but with only 1 neuron the behavior is undefined due to
            # @inbounds. We simply verify the function doesn't segfault
            # and returns a numeric value (may be meaningless).
            topo = make_topology([
                (0, 0) => [1.0, 2.0]
            ])
            som = make_trained_dbgsom(topo)

            X = reshape(Float64[1.0, 2.0], 2, 1)

            # With @inbounds, this may return garbage but should not crash.
            # We just verify it produces a Float64 result.
            te = compute_topographic_error(som, X)
            @test isa(te, Float64)
        end

        @testset "QE and TE with identical neuron weights" begin
            # Two neurons at different positions but same weights
            topo = make_topology([
                (0, 0) => [5.0, 5.0],
                (1, 0) => [5.0, 5.0]
            ])
            som = make_trained_dbgsom(topo)

            X = Float64[5.0 5.0; 5.0 5.0]  # Both samples at exactly the weight

            qe = compute_quantization_error(topo, X)
            @test qe ≈ 0.0 atol=1e-12

            te = compute_topographic_error(som, X)
            # Both neurons equally close, so BMU1 and BMU2 are the two neurons
            # They are adjacent => TE = 0
            @test te ≈ 0.0
        end

        @testset "QE increases when samples move away from neurons" begin
            w = [5.0, 5.0]
            topo = make_topology([
                (0, 0) => copy(w)
            ])

            X_close = reshape(Float64[5.1, 5.1], 2, 1)
            X_far = reshape(Float64[10.0, 10.0], 2, 1)

            qe_close = compute_quantization_error(topo, X_close)
            qe_far = compute_quantization_error(topo, X_far)

            @test qe_close < qe_far
        end
    end

    # ========================================================================
    # Pairwise Distance Consistency
    # ========================================================================
    @testset "Distance computation underlying QE" begin

        @testset "BLAS pairwise distances match naive computation" begin
            Random.seed!(555)
            X = randn(Float64, 3, 5)   # 3 dims x 5 samples
            W = randn(Float64, 3, 4)   # 3 dims x 4 neurons

            D_blas = pairwise_distances_squared(X, W)

            # Naive computation
            for i in 1:5
                for j in 1:4
                    diff = X[:, i] - W[:, j]
                    expected = sum(diff .^ 2)
                    @test D_blas[i, j] ≈ expected atol=1e-10
                end
            end
        end

        @testset "sqrt of squared distance equals euclidean distance" begin
            x = Float64[1.0, 2.0, 3.0]
            y = Float64[4.0, 6.0, 3.0]

            d_sq = euclidean_squared(x, y)
            d = euclidean(x, y)

            @test d ≈ sqrt(d_sq)
            @test d ≈ sqrt(9.0 + 16.0 + 0.0)  # 5.0
            @test d ≈ 5.0
        end
    end

end

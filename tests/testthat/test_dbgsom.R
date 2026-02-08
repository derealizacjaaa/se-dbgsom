# test_dbgsom.R - Tests for the DBGSOM R implementation
#
# Can be run standalone by sourcing all R files first, or via testthat.

# Source all R files if not loaded as package
if (!exists("dbgsom", mode = "function")) {
  r_dir <- file.path(dirname(dirname(getwd())), "R")
  if (!dir.exists(r_dir)) r_dir <- file.path(getwd(), "R")
  for (f in c("utils.R", "topology.R", "distance.R", "neighborhood.R",
              "bmu.R", "nan_handling.R", "batch_update.R", "growth.R",
              "dbgsom.R", "plotting.R", "clustering.R", "metrics.R", "pooled.R")) {
    source(file.path(r_dir, f))
  }
}

# ============================================================================
# Utility functions
# ============================================================================

test_that("pos_key and key_to_pos roundtrip", {
  expect_equal(pos_key(3, 7), "3,7")
  expect_equal(key_to_pos("3,7"), c(3L, 7L))
  expect_equal(pos_to_key(c(3, 7)), "3,7")
  expect_equal(key_to_pos(pos_to_key(c(-1, 5))), c(-1L, 5L))
})

test_that("new_neuron creates correct structure", {
  n <- new_neuron(c(1.0, 2.0, 3.0))
  expect_equal(n$weights, c(1.0, 2.0, 3.0))
  expect_equal(n$error, 0.0)
  expect_equal(n$hits, 0L)
})

# ============================================================================
# Topology
# ============================================================================

test_that("get_neighbors_hex returns 6 neighbors", {
  nbrs <- get_neighbors_hex(c(0L, 0L))
  expect_equal(length(nbrs), 6)

  nbrs_odd <- get_neighbors_hex(c(1L, 1L))
  expect_equal(length(nbrs_odd), 6)
})

test_that("grid_distance_hex is zero for same position", {
  expect_equal(grid_distance_hex(c(0L, 0L), c(0L, 0L)), 0)
})

test_that("grid_distance_hex is 1 for adjacent cells", {
  pos <- c(2L, 2L)
  nbrs <- get_neighbors_hex(pos)
  for (nbr in nbrs) {
    expect_equal(grid_distance_hex(pos, nbr), 1)
  }
})

test_that("grid_distance_hex symmetry", {
  expect_equal(grid_distance_hex(c(0L, 0L), c(3L, 2L)),
               grid_distance_hex(c(3L, 2L), c(0L, 0L)))
})

test_that("init_topology creates correct grid", {
  neurons <- init_topology(3L, 2L, 2L)
  expect_equal(length(neurons), 4)
  expect_true("0,0" %in% names(neurons))
  expect_true("1,1" %in% names(neurons))
  expect_equal(length(neurons[["0,0"]]$weights), 3)
})

test_that("is_boundary works for 2x2 grid", {
  neurons <- init_topology(3L, 2L, 2L)
  # All neurons in 2x2 grid are boundary
  for (key in names(neurons)) {
    pos <- key_to_pos(key)
    expect_true(is_boundary(neurons, pos))
  }
})

test_that("get_empty_neighbors returns correct count", {
  neurons <- init_topology(3L, 2L, 2L)
  # Corner neuron at (0,0) has some empty neighbors
  empty <- get_empty_neighbors(neurons, c(0L, 0L))
  expect_true(length(empty) > 0)
  expect_true(length(empty) < 6)
})

test_that("grid_bounds correct for 2x2 grid", {
  neurons <- init_topology(3L, 2L, 2L)
  b <- grid_bounds(neurons)
  expect_equal(b$x_min, 0L)
  expect_equal(b$x_max, 1L)
  expect_equal(b$y_min, 0L)
  expect_equal(b$y_max, 1L)
})

# ============================================================================
# Distance
# ============================================================================

test_that("euclidean_sq is correct", {
  expect_equal(euclidean_sq(c(0, 0), c(3, 4)), 25)
  expect_equal(euclidean_sq(c(1, 1), c(1, 1)), 0)
})

test_that("euclidean_dist is correct", {
  expect_equal(euclidean_dist(c(0, 0), c(3, 4)), 5)
})

test_that("euclidean_sq_na handles missing values", {
  expect_equal(euclidean_sq_na(c(1, NA, 3), c(2, 5, 4)),
               (1 + 1) * (3 / 2))  # 2 valid dims out of 3, scale by 3/2
  expect_equal(euclidean_sq_na(c(NA, NA), c(1, 2)), Inf)
})

test_that("pairwise_dist_sq correct for simple case", {
  X <- matrix(c(1, 0, 0, 1), nrow = 2)  # 2 features, 2 samples
  W <- matrix(c(0, 0, 1, 1), nrow = 2)  # 2 features, 2 neurons
  D <- pairwise_dist_sq(X, W)
  expect_equal(nrow(D), 2)
  expect_equal(ncol(D), 2)
  # D[1,1] = ||(1,0)-(0,0)||^2 = 1
  expect_equal(D[1, 1], 1)
  # D[2,2] = ||(0,1)-(1,1)||^2 = 1
  expect_equal(D[2, 2], 1)
})

# ============================================================================
# Neighborhood
# ============================================================================

test_that("gaussian_neighborhood at zero distance is 1", {
  expect_equal(gaussian_neighborhood(0, 1.0), 1.0)
})

test_that("gaussian_neighborhood decays with distance", {
  h1 <- gaussian_neighborhood(1, 1.0)
  h2 <- gaussian_neighborhood(2, 1.0)
  expect_true(h1 > h2)
  expect_true(h2 > 0)
})

test_that("decay_sigma is linear", {
  s <- decay_sigma(1, 10, 5.0, 1.0)
  expect_equal(s, 5.0)  # First epoch = start
  s_end <- decay_sigma(10, 10, 5.0, 1.0)
  expect_equal(s_end, 1.0)  # Last epoch = end
})

# ============================================================================
# Growth threshold
# ============================================================================

test_that("spreading_factor threshold is correct", {
  sf <- spreading_factor(0.5)
  gt <- compute_growth_threshold(sf, 4L)
  expect_equal(gt, -log(0.5) * 4)
})

test_that("statistical_error threshold requires data", {
  se <- statistical_error(1.0)
  expect_error(compute_growth_threshold(se, 4L))
})

test_that("statistical_error threshold computes from data", {
  se <- statistical_error(1.0)
  X <- matrix(rnorm(40), nrow = 4, ncol = 10)
  gt <- compute_growth_threshold(se, 4L, X)
  expect_true(gt > 0)
})

# ============================================================================
# DBGSOM creation and training
# ============================================================================

test_that("dbgsom constructor works", {
  som <- dbgsom(input_dim = 4)
  expect_equal(som$input_dim, 4)
  expect_false(som$trained)
  expect_equal(length(som$neurons), 4)  # 2x2 grid
  expect_s3_class(som, "dbgsom")
})

test_that("dbgsom fits on iris data", {
  set.seed(42)
  data(iris)
  X <- as.matrix(iris[, 1:4])
  X <- scale(X)

  som <- dbgsom(input_dim = 4,
                gt_method = statistical_error(1.5),
                max_neurons = 30,
                n_iter = 20)
  som <- fit(som, X)

  expect_true(som$trained)
  expect_true(length(som$neurons) > 4)  # Should have grown
  expect_true(length(som$neurons) <= 30)  # Should respect max
})

test_that("predict returns correct number of BMUs", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)

  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)

  bmus <- predict(som, X)
  expect_equal(length(bmus), 25)
  # All BMUs should be valid neuron keys
  for (key in bmus) {
    expect_true(key %in% names(som$neurons))
  }
})

test_that("transform returns coordinate matrix", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)

  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)

  coords <- transform(som, X)
  expect_equal(nrow(coords), 25)
  expect_equal(ncol(coords), 2)
  expect_equal(colnames(coords), c("x", "y"))
})

# ============================================================================
# NA handling
# ============================================================================

test_that("compute_missingness_info correct", {
  X <- matrix(c(1, NA, 3, 4, 5, NA), nrow = 2, ncol = 3)
  info <- compute_missingness_info(X)
  expect_equal(length(info$rates), 2)
  expect_true(all(info$rates >= 0 & info$rates <= 1))
  expect_equal(length(info$obs_weights), 2)
})

test_that("dbgsom handles NA data", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  # Introduce 10% NAs
  na_idx <- sample(length(X), size = 20)
  X[na_idx] <- NA

  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)

  expect_true(som$trained)
  expect_true(!is.null(som$missingness_info))
})

# ============================================================================
# Visualization
# ============================================================================

test_that("compute_umatrix returns values for all neurons", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)

  umat <- compute_umatrix(som)
  expect_equal(length(umat), length(som$neurons))
  expect_true(all(umat >= 0))
})

test_that("compute_hit_counts sums to n_samples", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)

  hits <- compute_hit_counts(som, X)
  expect_equal(sum(hits), 25)
})

test_that("compute_component_planes returns one per dimension", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)

  planes <- compute_component_planes(som)
  expect_equal(length(planes), 4)
  for (p in planes) {
    expect_equal(length(p), length(som$neurons))
  }
})

# ============================================================================
# Clustering
# ============================================================================

test_that("vesanto_clustering ward method works", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)

  clust <- vesanto_clustering(som, method = "ward")
  expect_true(clust$n_clusters >= 2)
  expect_equal(length(clust$labels), length(som$neurons))
  expect_true(!is.null(clust$hclust_obj))
})

test_that("vesanto_clustering kmeans method works", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)

  clust <- vesanto_clustering(som, method = "kmeans", seed = 42)
  expect_true(clust$n_clusters >= 2)
})

test_that("vesanto_clustering with fixed k", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)

  clust <- vesanto_clustering(som, method = "ward", n_clusters = 3)
  expect_equal(clust$n_clusters, 3)
})

test_that("assign_cluster_labels maps all samples", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)

  clust <- vesanto_clustering(som, method = "ward", n_clusters = 3)
  sample_labels <- assign_cluster_labels(som, X, clust$labels)
  expect_equal(length(sample_labels), 50)
  expect_true(all(sample_labels %in% 1:3))
})

# ============================================================================
# Pooled preprocessing
# ============================================================================

test_that("pooled_prepare normalizes data", {
  pooled <- pooled_prepare(iris[, 1:4])
  expect_equal(nrow(pooled$X), 150)
  expect_equal(ncol(pooled$X), 4)
  # Should be approximately mean 0, sd 1
  for (j in seq_len(ncol(pooled$X))) {
    expect_true(abs(mean(pooled$X[, j])) < 0.01)
    expect_true(abs(sd(pooled$X[, j]) - 1.0) < 0.01)
  }
})

test_that("pooled_prepare with entity/time columns", {
  df <- data.frame(
    country = rep(c("A", "B"), each = 5),
    year = rep(2015:2019, 2),
    gdp = rnorm(10),
    pop = rnorm(10)
  )
  pooled <- pooled_prepare(df, entity_col = "country", time_col = "year")
  expect_equal(ncol(pooled$X), 2)
  expect_equal(nrow(pooled$X), 10)
  expect_equal(pooled$feature_names, c("gdp", "pop"))
  expect_equal(length(pooled$entity_ids), 10)
  expect_equal(length(pooled$time_ids), 10)
})

test_that("pooled_prepare without normalization", {
  pooled <- pooled_prepare(iris[, 1:4], normalize = FALSE)
  expect_equal(pooled$X[1, 1], iris$Sepal.Length[1])
})

# ============================================================================
# Metrics
# ============================================================================

test_that("compute_topographic_error in [0, 1]", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)

  te <- compute_topographic_error(som, X)
  expect_true(te >= 0 && te <= 1)
})

test_that("cluster_purity in [0, 1]", {
  set.seed(42)
  data(iris)
  X <- scale(as.matrix(iris[, 1:4]))

  som <- dbgsom(input_dim = 4, gt_method = statistical_error(1.5),
                max_neurons = 30, n_iter = 20)
  som <- fit(som, X)

  clust <- vesanto_clustering(som, method = "ward", n_clusters = 3)
  bmu_keys <- predict(som, X)
  purity <- cluster_purity(bmu_keys, clust$labels, as.integer(iris$Species))
  expect_true(purity >= 0 && purity <= 1)
})

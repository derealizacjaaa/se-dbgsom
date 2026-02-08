# Standalone test runner - source all files and run tests
r_dir <- file.path(getwd(), "R")
for (f in c("utils.R", "topology.R", "distance.R", "neighborhood.R",
            "bmu.R", "nan_handling.R", "batch_update.R", "growth.R",
            "dbgsom.R", "plotting.R", "clustering.R", "metrics.R", "pooled.R")) {
  source(file.path(r_dir, f))
}

cat("=== Running DBGSOM R tests ===\n\n")
errors <- 0L
tests <- 0L

run_test <- function(name, expr) {
  tests <<- tests + 1L
  result <- tryCatch({
    expr
    cat(sprintf("  PASS: %s\n", name))
    TRUE
  }, error = function(e) {
    cat(sprintf("  FAIL: %s - %s\n", name, e$message))
    errors <<- errors + 1L
    FALSE
  })
}

# --- Utility tests ---
cat("Utilities:\n")
run_test("pos_key roundtrip", {
  stopifnot(pos_key(3, 7) == "3,7")
  stopifnot(all(key_to_pos("3,7") == c(3L, 7L)))
})

run_test("new_neuron", {
  n <- new_neuron(c(1, 2, 3))
  stopifnot(all(n$weights == c(1, 2, 3)))
  stopifnot(n$error == 0 && n$hits == 0L)
})

# --- Topology tests ---
cat("\nTopology:\n")
run_test("6 hex neighbors", {
  stopifnot(length(get_neighbors_hex(c(0L, 0L))) == 6)
  stopifnot(length(get_neighbors_hex(c(1L, 1L))) == 6)
})

run_test("grid distance self = 0", {
  stopifnot(grid_distance_hex(c(0L, 0L), c(0L, 0L)) == 0)
})

run_test("grid distance adjacent = 1", {
  pos <- c(2L, 2L)
  for (nbr in get_neighbors_hex(pos)) {
    stopifnot(grid_distance_hex(pos, nbr) == 1)
  }
})

run_test("grid distance symmetry", {
  stopifnot(grid_distance_hex(c(0L, 0L), c(3L, 2L)) ==
            grid_distance_hex(c(3L, 2L), c(0L, 0L)))
})

run_test("init_topology 2x2", {
  neurons <- init_topology(3L, 2L, 2L)
  stopifnot(length(neurons) == 4)
  stopifnot("0,0" %in% names(neurons))
  stopifnot(length(neurons[["0,0"]]$weights) == 3)
})

run_test("all 2x2 neurons are boundary", {
  neurons <- init_topology(3L, 2L, 2L)
  for (key in names(neurons)) {
    stopifnot(is_boundary(neurons, key_to_pos(key)))
  }
})

# --- Distance tests ---
cat("\nDistance:\n")
run_test("euclidean_sq", {
  stopifnot(euclidean_sq(c(0, 0), c(3, 4)) == 25)
  stopifnot(euclidean_dist(c(0, 0), c(3, 4)) == 5)
})

run_test("euclidean_sq_na scaling", {
  d <- euclidean_sq_na(c(1, NA, 3), c(2, 5, 4))
  stopifnot(abs(d - (1 + 1) * (3 / 2)) < 1e-10)
  stopifnot(euclidean_sq_na(c(NA, NA), c(1, 2)) == Inf)
})

run_test("pairwise_dist_sq", {
  X <- matrix(c(1, 0, 0, 1), nrow = 2)
  W <- matrix(c(0, 0, 1, 1), nrow = 2)
  D <- pairwise_dist_sq(X, W)
  stopifnot(nrow(D) == 2 && ncol(D) == 2)
  stopifnot(D[1, 1] == 1)
  stopifnot(D[2, 2] == 1)
})

# --- Neighborhood tests ---
cat("\nNeighborhood:\n")
run_test("gaussian at zero = 1", {
  stopifnot(gaussian_neighborhood(0, 1.0) == 1.0)
})

run_test("gaussian decays", {
  stopifnot(gaussian_neighborhood(1, 1.0) > gaussian_neighborhood(2, 1.0))
})

run_test("sigma decay linear", {
  stopifnot(decay_sigma(1, 10, 5.0, 1.0) == 5.0)
  stopifnot(decay_sigma(10, 10, 5.0, 1.0) == 1.0)
})

# --- Growth threshold tests ---
cat("\nGrowth Threshold:\n")
run_test("spreading_factor", {
  sf <- spreading_factor(0.5)
  gt <- compute_growth_threshold(sf, 4L)
  stopifnot(abs(gt - (-log(0.5) * 4)) < 1e-10)
})

run_test("statistical_error with data", {
  se <- statistical_error(1.0)
  set.seed(42)
  X <- matrix(rnorm(40), nrow = 4, ncol = 10)
  gt <- compute_growth_threshold(se, 4L, X)
  stopifnot(gt > 0)
})

# --- DBGSOM core tests ---
cat("\nDBGSOM core:\n")
run_test("constructor", {
  som <- dbgsom(input_dim = 4)
  stopifnot(som$input_dim == 4)
  stopifnot(!som$trained)
  stopifnot(length(som$neurons) == 4)
  stopifnot(inherits(som, "dbgsom"))
})

run_test("fit on random data", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  stopifnot(som$trained)
  stopifnot(length(som$neurons) >= 4)
})

run_test("fit on iris", {
  set.seed(42)
  data(iris)
  X <- scale(as.matrix(iris[, 1:4]))
  som <- dbgsom(input_dim = 4, gt_method = statistical_error(0.5),
                max_neurons = 50, n_iter = 40)
  som <- fit(som, X)
  stopifnot(som$trained)
  stopifnot(length(som$neurons) > 4)
  stopifnot(length(som$neurons) <= 50)
})

run_test("predict", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  bmus <- predict(som, X)
  stopifnot(length(bmus) == 25)
  for (key in bmus) stopifnot(key %in% names(som$neurons))
})

run_test("transform", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  coords <- transform(som, X)
  stopifnot(nrow(coords) == 25)
  stopifnot(ncol(coords) == 2)
  stopifnot(all(colnames(coords) == c("x", "y")))
})

# --- NA handling tests ---
cat("\nNA handling:\n")
run_test("missingness_info", {
  X <- matrix(c(1, NA, 3, 4, 5, NA), nrow = 2, ncol = 3)
  info <- compute_missingness_info(X)
  stopifnot(length(info$rates) == 2)
  stopifnot(all(info$rates >= 0 & info$rates <= 1))
})

run_test("fit with NAs", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  X[sample(length(X), 20)] <- NA
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  stopifnot(som$trained)
  stopifnot(!is.null(som$missingness_info))
})

# --- Visualization tests ---
cat("\nVisualization:\n")
run_test("umatrix", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  umat <- compute_umatrix(som)
  stopifnot(length(umat) == length(som$neurons))
  stopifnot(all(umat >= 0))
})

run_test("hit_counts sum to n_samples", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  hits <- compute_hit_counts(som, X)
  stopifnot(sum(hits) == 25)
})

run_test("component_planes count", {
  set.seed(42)
  X <- matrix(rnorm(100), nrow = 25, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 20, n_iter = 10)
  som <- fit(som, X)
  planes <- compute_component_planes(som)
  stopifnot(length(planes) == 4)
})

# --- Clustering tests ---
cat("\nClustering:\n")
run_test("vesanto ward", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)
  clust <- vesanto_clustering(som, method = "ward")
  stopifnot(clust$n_clusters >= 2)
  stopifnot(length(clust$labels) == length(som$neurons))
})

run_test("vesanto kmeans", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)
  clust <- vesanto_clustering(som, method = "kmeans", seed = 42)
  stopifnot(clust$n_clusters >= 2)
})

run_test("vesanto fixed k", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)
  clust <- vesanto_clustering(som, method = "ward", n_clusters = 3)
  stopifnot(clust$n_clusters == 3)
})

run_test("assign_cluster_labels", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)
  clust <- vesanto_clustering(som, method = "ward", n_clusters = 3)
  sample_labels <- assign_cluster_labels(som, X, clust$labels)
  stopifnot(length(sample_labels) == 50)
  stopifnot(all(sample_labels %in% 1:3))
})

# --- Pooled preprocessing tests ---
cat("\nPooled preprocessing:\n")
run_test("pooled_prepare normalization", {
  pooled <- pooled_prepare(iris[, 1:4])
  stopifnot(nrow(pooled$X) == 150)
  stopifnot(ncol(pooled$X) == 4)
  for (j in 1:4) {
    stopifnot(abs(mean(pooled$X[, j])) < 0.01)
    stopifnot(abs(sd(pooled$X[, j]) - 1.0) < 0.01)
  }
})

run_test("pooled_prepare entity/time", {
  df <- data.frame(
    country = rep(c("A", "B"), each = 5),
    year = rep(2015:2019, 2),
    gdp = rnorm(10),
    pop = rnorm(10)
  )
  pooled <- pooled_prepare(df, entity_col = "country", time_col = "year")
  stopifnot(ncol(pooled$X) == 2)
  stopifnot(nrow(pooled$X) == 10)
  stopifnot(all(pooled$feature_names == c("gdp", "pop")))
  stopifnot(length(pooled$entity_ids) == 10)
})

# --- Metrics tests ---
cat("\nMetrics:\n")
run_test("topographic_error in [0,1]", {
  set.seed(42)
  X <- matrix(rnorm(200), nrow = 50, ncol = 4)
  som <- dbgsom(input_dim = 4, max_neurons = 30, n_iter = 20)
  som <- fit(som, X)
  te <- compute_topographic_error(som, X)
  stopifnot(te >= 0 && te <= 1)
})

run_test("cluster_purity in [0,1]", {
  set.seed(42)
  data(iris)
  X <- scale(as.matrix(iris[, 1:4]))
  som <- dbgsom(input_dim = 4, gt_method = statistical_error(1.5),
                max_neurons = 30, n_iter = 20)
  som <- fit(som, X)
  clust <- vesanto_clustering(som, method = "ward", n_clusters = 3)
  bmu_keys <- predict(som, X)
  purity <- cluster_purity(bmu_keys, clust$labels, as.integer(iris$Species))
  stopifnot(purity >= 0 && purity <= 1)
})

# === Summary ===
cat(sprintf("\n=== Results: %d/%d tests passed ===\n", tests - errors, tests))
if (errors > 0) {
  cat(sprintf("FAILURES: %d\n", errors))
  quit(status = 1)
} else {
  cat("All tests passed!\n")
}

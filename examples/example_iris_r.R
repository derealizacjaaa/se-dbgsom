# example_iris_r.R - DBGSOM demonstration with iris dataset
#
# Demonstrates the full workflow:
# 1. Data preparation with pooled model
# 2. DBGSOM training
# 3. Quality metrics
# 4. Vesanto two-level clustering
# 5. Evaluation against ground truth

# Source all R files (when not installed as package)
r_dir <- file.path(dirname(getwd()), "R")
if (!dir.exists(r_dir)) r_dir <- file.path(getwd(), "R")
for (f in c("utils.R", "topology.R", "distance.R", "neighborhood.R",
            "bmu.R", "nan_handling.R", "batch_update.R", "growth.R",
            "dbgsom.R", "plotting.R", "clustering.R", "metrics.R", "pooled.R")) {
  source(file.path(r_dir, f))
}

cat("=== DBGSOM R Implementation Demo (Iris Dataset) ===\n\n")

# ============================================================================
# 1. Load and prepare data
# ============================================================================
data(iris)

pooled <- pooled_prepare(iris,
                         feature_cols = c("Sepal.Length", "Sepal.Width",
                                         "Petal.Length", "Petal.Width"),
                         normalize = TRUE)

cat(sprintf("Data: %d samples x %d features\n", nrow(pooled$X), ncol(pooled$X)))
cat(sprintf("Features: %s\n\n", paste(pooled$feature_names, collapse = ", ")))

# ============================================================================
# 2. Create and train DBGSOM
# ============================================================================
set.seed(42)

som <- dbgsom(
  input_dim = ncol(pooled$X),
  gt_method = statistical_error(lambda = 1.0),
  max_neurons = 80,
  n_iter = 100
)

cat("Before training:\n")
print(som)
cat("\n")

som <- fit(som, pooled$X)

cat("After training:\n")
print(som)
cat("\n")

# ============================================================================
# 3. Quality metrics
# ============================================================================
cat("--- Quality Metrics ---\n")
Xc <- ensure_col_major(pooled$X, som$input_dim)
qe <- compute_quantization_error(som$neurons, Xc)
te <- compute_topographic_error(som, pooled$X)
cat(sprintf("Quantization Error:  %.4f\n", qe))
cat(sprintf("Topographic Error:   %.4f\n", te))
cat("\n")

# ============================================================================
# 4. Visualization data
# ============================================================================
cat("--- Grid ---\n")
cat(format_grid_ascii(som), "\n\n")

umat <- compute_umatrix(som)
cat(sprintf("U-matrix: min=%.3f, mean=%.3f, max=%.3f\n",
            min(umat), mean(umat), max(umat)))

hits <- compute_hit_counts(som, pooled$X)
cat(sprintf("Hit counts: min=%d, mean=%.1f, max=%d\n",
            min(hits), mean(hits), max(hits)))
cat(sprintf("Active neurons: %d / %d\n\n", sum(hits > 0), length(hits)))

# ============================================================================
# 5. Vesanto clustering
# ============================================================================
cat("--- Vesanto Clustering ---\n")
clust <- vesanto_clustering(som, method = "ward", auto_k_method = "davies_bouldin",
                             min_clusters = 2, max_clusters = 10)
cat(sprintf("Optimal k: %d\n", clust$optimal_k))
cat(sprintf("Clusters found: %d\n", clust$n_clusters))

if (length(clust$db_scores) > 0) {
  cat("Davies-Bouldin scores:\n")
  for (k in names(clust$db_scores)) {
    cat(sprintf("  k=%s: DB=%.3f, Sil=%.3f\n",
                k, clust$db_scores[k], clust$sil_scores[k]))
  }
}
cat("\n")

# ============================================================================
# 6. Evaluation against ground truth
# ============================================================================
cat("--- Evaluation ---\n")
true_labels <- as.integer(iris$Species)
sample_labels <- assign_cluster_labels(som, pooled$X, clust$labels)
purity <- cluster_purity(predict(som, pooled$X), clust$labels, true_labels)
cat(sprintf("Purity: %.4f\n", purity))

cat("\nSummary:\n")
cat(format_summary(som, pooled$X), "\n")

cat("\n=== Demo complete ===\n")

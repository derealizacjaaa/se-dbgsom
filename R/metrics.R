# metrics.R - Quality and clustering evaluation metrics

# ============================================================================
# Topographic Error
# ============================================================================

#' Compute topographic error
#'
#' Proportion of samples where BMU1 and BMU2 are not adjacent on the grid.
#' 0 = perfect topology preservation, 1 = worst.
#'
#' @param som trained dbgsom object
#' @param X matrix (n_samples x input_dim)
#' @return numeric in [0, 1]
compute_topographic_error <- function(som, X) {
  X <- ensure_col_major(X, som$input_dim)
  res <- find_two_bmus(som$neurons, X)

  n_samples <- length(res$bmu1_keys)
  n_errors <- 0L

  # Precompute neighbor sets
  neighbor_sets <- list()
  for (key in names(som$neurons)) {
    pos <- key_to_pos(key)
    nbrs <- get_neighbors_hex(pos)
    neighbor_sets[[key]] <- vapply(nbrs, pos_to_key, character(1))
  }

  for (i in seq_len(n_samples)) {
    if (!(res$bmu2_keys[i] %in% neighbor_sets[[res$bmu1_keys[i]]])) {
      n_errors <- n_errors + 1L
    }
  }

  n_errors / n_samples
}

# ============================================================================
# Clustering Metrics (against ground truth)
# ============================================================================

#' Compute cluster purity
#'
#' Fraction of samples matching the dominant class in each cluster.
#'
#' @param bmu_keys character vector of BMU position keys
#' @param neuron_labels named integer vector of neuron cluster labels
#' @param true_labels vector of ground truth class labels
#' @return numeric purity in [0, 1]
cluster_purity <- function(bmu_keys, neuron_labels, true_labels) {
  pred_labels <- neuron_labels[bmu_keys]
  clusters <- unique(pred_labels)
  total_correct <- 0L

  for (cl in clusters) {
    members <- which(pred_labels == cl)
    if (length(members) > 0) {
      tab <- table(true_labels[members])
      total_correct <- total_correct + max(tab)
    }
  }

  total_correct / length(true_labels)
}

#' Compute all evaluation metrics for a clustering
#'
#' @param som trained dbgsom object
#' @param X matrix (n_samples x input_dim)
#' @param neuron_labels named integer vector of neuron cluster labels
#' @param true_labels vector of ground truth class labels
#' @return list with silhouette, purity, n_clusters, qe, te
compute_all_metrics <- function(som, X, neuron_labels, true_labels) {
  Xc <- ensure_col_major(X, som$input_dim)
  bmu_keys <- predict.dbgsom(som, X)

  # Neuron-level silhouette
  nw <- get_neuron_weights(som)
  keys <- names(som$neurons)
  labels_vec <- neuron_labels[keys]
  sil_score <- 0.0
  if (length(unique(labels_vec)) >= 2) {
    d <- dist(nw$weights)
    sil <- cluster::silhouette(labels_vec, d)
    sil_score <- mean(sil[, "sil_width"])
  }

  list(
    silhouette = sil_score,
    purity = cluster_purity(bmu_keys, neuron_labels, true_labels),
    n_clusters = length(unique(neuron_labels)),
    qe = compute_quantization_error(som$neurons, Xc),
    te = compute_topographic_error(som, X)
  )
}

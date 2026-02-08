# clustering.R - Vesanto-Alhoniemi two-level clustering
#
# Based on: "Clustering of the Self-Organizing Map"
# Vesanto & Alhoniemi (IEEE Trans. Neural Networks, 2000)
#
# Two-level approach:
# 1. SOM training produces prototypes (neuron weight vectors)
# 2. Cluster prototypes using Ward's hierarchical or K-means
# 3. Determine optimal k using Davies-Bouldin index or silhouette

#' Vesanto-Alhoniemi two-level clustering on SOM prototypes
#'
#' @param som trained dbgsom object
#' @param method character: "ward" (default) or "kmeans"
#' @param n_clusters integer or NULL for auto-detection
#' @param auto_k_method character: "davies_bouldin" or "silhouette"
#' @param min_clusters integer minimum k to try (default 2)
#' @param max_clusters integer maximum k to try (default 15)
#' @param seed integer random seed for kmeans (optional)
#' @return list with:
#'   labels: named integer vector (pos_key -> cluster id, 1-indexed)
#'   n_clusters: integer number of clusters
#'   optimal_k: integer selected k
#'   hclust_obj: hclust object (for ward method, NULL otherwise)
#'   db_scores: named numeric (k -> DB index)
#'   sil_scores: named numeric (k -> silhouette score)
vesanto_clustering <- function(som,
                                method = "ward",
                                n_clusters = NULL,
                                auto_k_method = "davies_bouldin",
                                min_clusters = 2L,
                                max_clusters = 15L,
                                seed = NULL) {
  stopifnot(method %in% c("ward", "kmeans"))
  stopifnot(auto_k_method %in% c("davies_bouldin", "silhouette"))
  stopifnot(min_clusters >= 2L)
  stopifnot(is.null(max_clusters) || max_clusters >= min_clusters)

  nw <- get_neuron_weights(som)
  positions <- nw$positions
  W <- nw$weights  # n_neurons x input_dim
  n_neurons <- nrow(W)
  keys <- names(som$neurons)

  # Edge cases
  if (n_neurons < 2) {
    labels <- setNames(rep(1L, length(keys)), keys)
    return(list(labels = labels, n_clusters = 1L, optimal_k = 1L,
                hclust_obj = NULL, db_scores = numeric(0), sil_scores = numeric(0)))
  }

  if (!is.null(n_clusters) && n_clusters > n_neurons) {
    n_clusters <- n_neurons
  }

  max_k <- min(max_clusters, n_neurons - 1L)
  db_scores <- numeric(0)
  sil_scores <- numeric(0)

  if (method == "ward") {
    result <- vesanto_ward(W, keys, n_clusters, min_clusters, max_k,
                            auto_k_method, db_scores, sil_scores)
  } else {
    result <- vesanto_kmeans(W, keys, n_clusters, min_clusters, max_k,
                              auto_k_method, db_scores, sil_scores, seed)
  }

  result
}

#' Ward's hierarchical clustering on SOM prototypes
vesanto_ward <- function(W, keys, n_clusters, min_k, max_k,
                          auto_k_method, db_scores, sil_scores) {
  d <- dist(W, method = "euclidean")
  hc <- hclust(d, method = "ward.D2")

  if (is.null(n_clusters)) {
    # Evaluate k range
    for (k in min_k:max_k) {
      cl <- cutree(hc, k = k)
      if (length(unique(cl)) < 2) next

      scores <- tryCatch(
        compute_internal_scores(W, cl),
        error = function(e) NULL
      )
      if (!is.null(scores)) {
        db_scores[as.character(k)] <- scores$db
        sil_scores[as.character(k)] <- scores$sil
      }
    }

    n_clusters <- select_optimal_k(db_scores, sil_scores, auto_k_method, min_k)
  }

  cl <- cutree(hc, k = n_clusters)
  labels <- setNames(as.integer(cl), keys)

  list(labels = labels, n_clusters = n_clusters, optimal_k = n_clusters,
       hclust_obj = hc, db_scores = db_scores, sil_scores = sil_scores)
}

#' K-means clustering on SOM prototypes
vesanto_kmeans <- function(W, keys, n_clusters, min_k, max_k,
                            auto_k_method, db_scores, sil_scores, seed) {
  if (is.null(n_clusters)) {
    for (k in min_k:max_k) {
      if (!is.null(seed)) set.seed(seed)
      km <- kmeans(W, centers = k, nstart = 10)
      cl <- km$cluster
      if (length(unique(cl)) < 2) next

      scores <- tryCatch(
        compute_internal_scores(W, cl),
        error = function(e) NULL
      )
      if (!is.null(scores)) {
        db_scores[as.character(k)] <- scores$db
        sil_scores[as.character(k)] <- scores$sil
      }
    }

    n_clusters <- select_optimal_k(db_scores, sil_scores, auto_k_method, min_k)
  }

  if (!is.null(seed)) set.seed(seed)
  km <- kmeans(W, centers = n_clusters, nstart = 10)
  labels <- setNames(as.integer(km$cluster), keys)

  list(labels = labels, n_clusters = n_clusters, optimal_k = n_clusters,
       hclust_obj = NULL, db_scores = db_scores, sil_scores = sil_scores)
}

#' Compute internal clustering scores (Davies-Bouldin and silhouette)
#' @param W matrix of weight vectors (n x p)
#' @param cl integer vector of cluster assignments
#' @return list with db and sil scores
compute_internal_scores <- function(W, cl) {
  d <- dist(W)
  sil <- cluster::silhouette(cl, d)
  sil_score <- mean(sil[, "sil_width"])

  db_score <- davies_bouldin_index(W, cl)

  list(db = db_score, sil = sil_score)
}

#' Davies-Bouldin index (lower is better)
#' @param W matrix (n x p)
#' @param cl integer cluster assignments
#' @return numeric DB index
davies_bouldin_index <- function(W, cl) {
  clusters <- sort(unique(cl))
  k <- length(clusters)
  if (k < 2) return(Inf)

  # Compute cluster centroids and scatter (mean distance to centroid)
  centroids <- list()
  scatter <- numeric(k)
  for (i in seq_along(clusters)) {
    members <- which(cl == clusters[i])
    if (length(members) == 1) {
      centroids[[i]] <- W[members, , drop = FALSE]
      scatter[i] <- 0.0
    } else {
      cent <- colMeans(W[members, , drop = FALSE])
      centroids[[i]] <- matrix(cent, nrow = 1)
      scatter[i] <- mean(sqrt(rowSums(sweep(W[members, , drop = FALSE], 2, cent)^2)))
    }
  }

  # DB = (1/k) * sum_i max_{j!=i} (S_i + S_j) / d(C_i, C_j)
  db_vals <- numeric(k)
  for (i in seq_len(k)) {
    max_ratio <- 0.0
    for (j in seq_len(k)) {
      if (i == j) next
      d_ij <- sqrt(sum((centroids[[i]] - centroids[[j]])^2))
      if (d_ij > 0) {
        ratio <- (scatter[i] + scatter[j]) / d_ij
        if (ratio > max_ratio) max_ratio <- ratio
      }
    }
    db_vals[i] <- max_ratio
  }

  mean(db_vals)
}

#' Select optimal k from score profiles
select_optimal_k <- function(db_scores, sil_scores, method, fallback) {
  if (length(db_scores) == 0) return(fallback)

  if (method == "davies_bouldin") {
    as.integer(names(which.min(db_scores)))
  } else {
    as.integer(names(which.max(sil_scores)))
  }
}

#' Assign cluster labels to samples via their BMU's cluster
#'
#' @param som trained dbgsom object
#' @param X matrix (n_samples x input_dim)
#' @param neuron_labels named integer vector of neuron cluster labels
#' @return integer vector of sample cluster labels
assign_cluster_labels <- function(som, X, neuron_labels) {
  bmu_keys <- predict.dbgsom(som, X)
  neuron_labels[bmu_keys]
}

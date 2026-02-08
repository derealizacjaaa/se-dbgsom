# bmu.R - Best Matching Unit finding for DBGSOM

#' Find BMU index from a precomputed distance row
#' @param d_row numeric vector of squared distances
#' @return integer index of minimum
find_bmu_idx <- function(d_row) {
  which.min(d_row)
}

#' Find BMUs for all samples using precomputed distance matrix
#'
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @return character vector of position keys (one per sample)
find_bmus <- function(neurons, X) {
  wm <- get_weights_matrix(neurons)
  D <- pairwise_dist_sq_auto(X, wm$W)
  n_samples <- ncol(X)
  bmu_keys <- character(n_samples)
  for (i in seq_len(n_samples)) {
    idx <- find_bmu_idx(D[i, ])
    bmu_keys[i] <- wm$pos_keys[idx]
  }
  bmu_keys
}

#' Find BMUs using missingness-weighted distances
#'
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param miss_info missingness info list (or NULL)
#' @return character vector of position keys
find_bmus_weighted <- function(neurons, X, miss_info) {
  wm <- get_weights_matrix(neurons)
  if (!is.null(miss_info) && has_na(X)) {
    D <- pairwise_dist_sq_na_weighted(X, wm$W, miss_info)
  } else {
    D <- pairwise_dist_sq_auto(X, wm$W)
  }
  n_samples <- ncol(X)
  bmu_keys <- character(n_samples)
  for (i in seq_len(n_samples)) {
    bmu_keys[i] <- wm$pos_keys[find_bmu_idx(D[i, ])]
  }
  bmu_keys
}

#' Dispatch BMU finding: use weighted if miss_info available, else standard
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param miss_info missingness info list or NULL
#' @return character vector of position keys
dispatch_find_bmus <- function(neurons, X, miss_info) {
  if (!is.null(miss_info)) {
    find_bmus_weighted(neurons, X, miss_info)
  } else {
    find_bmus(neurons, X)
  }
}

#' Find BMUs and their distances for all samples
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @return list with bmu_keys (character) and distances (numeric)
find_bmus_with_distances <- function(neurons, X) {
  wm <- get_weights_matrix(neurons)
  D <- pairwise_dist_sq_auto(X, wm$W)
  n_samples <- ncol(X)
  bmu_keys <- character(n_samples)
  distances <- numeric(n_samples)
  for (i in seq_len(n_samples)) {
    idx <- find_bmu_idx(D[i, ])
    bmu_keys[i] <- wm$pos_keys[idx]
    distances[i] <- sqrt(D[i, idx])
  }
  list(bmu_keys = bmu_keys, distances = distances)
}

#' Find first and second BMUs for all samples (for topographic error)
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @return list with bmu1_keys, bmu2_keys, dist1s, dist2s
find_two_bmus <- function(neurons, X) {
  wm <- get_weights_matrix(neurons)
  D <- pairwise_dist_sq_auto(X, wm$W)
  n_samples <- ncol(X)
  n_neurons <- ncol(D)

  bmu1_keys <- character(n_samples)
  bmu2_keys <- character(n_samples)
  dist1s <- numeric(n_samples)
  dist2s <- numeric(n_samples)

  for (i in seq_len(n_samples)) {
    row <- D[i, ]
    # First minimum
    idx1 <- which.min(row)
    # Second minimum (exclude first)
    row2 <- row
    row2[idx1] <- Inf
    idx2 <- which.min(row2)

    bmu1_keys[i] <- wm$pos_keys[idx1]
    bmu2_keys[i] <- wm$pos_keys[idx2]
    dist1s[i] <- sqrt(row[idx1])
    dist2s[i] <- sqrt(row[idx2])
  }

  list(bmu1_keys = bmu1_keys, bmu2_keys = bmu2_keys,
       dist1s = dist1s, dist2s = dist2s)
}

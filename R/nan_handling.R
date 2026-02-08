# nan_handling.R - NA-aware distances and neighbor-based imputation
#
# All functions that modify neurons RETURN the modified list.

#' Compute per-dimension missingness rates and observation weights
#'
#' Observation weight for dimension d is (1 - rate_d)^2.
#'
#' @param X matrix (input_dim x n_samples)
#' @return list with rates, obs_weights, n_samples
compute_missingness_info <- function(X) {
  input_dim <- nrow(X)
  n_samples <- ncol(X)
  stopifnot(input_dim > 0, n_samples > 0)

  rates <- numeric(input_dim)
  for (d in seq_len(input_dim)) {
    rates[d] <- sum(is.na(X[d, ])) / n_samples
  }

  obs_weights <- (1.0 - rates)^2

  list(rates = rates, obs_weights = obs_weights, n_samples = n_samples)
}

#' Squared Euclidean distance weighted by per-variable observation rates
#'
#' Returns Inf if no valid dimensions.
#'
#' @param x numeric vector
#' @param y numeric vector
#' @param info missingness info list
#' @return scalar squared distance
euclidean_sq_na_weighted <- function(x, y, info) {
  dist_val <- 0.0
  used_weight <- 0.0
  total_weight <- sum(info$obs_weights)

  for (i in seq_along(x)) {
    w <- info$obs_weights[i]
    if (!is.na(x[i]) && !is.na(y[i]) && w > 0) {
      diff <- x[i] - y[i]
      dist_val <- dist_val + w * diff * diff
      used_weight <- used_weight + w
    }
  }

  if (used_weight > 0 && total_weight > 0) {
    dist_val * (total_weight / used_weight)
  } else {
    Inf
  }
}

#' Pairwise squared distances using missingness-weighted metric
#' @param X matrix (input_dim x n_samples)
#' @param W matrix (input_dim x n_neurons)
#' @param info missingness info list
#' @return matrix (n_samples x n_neurons)
pairwise_dist_sq_na_weighted <- function(X, W, info) {
  n_samples <- ncol(X)
  n_neurons <- ncol(W)
  D <- matrix(NA_real_, nrow = n_samples, ncol = n_neurons)
  for (j in seq_len(n_neurons)) {
    w <- W[, j]
    for (i in seq_len(n_samples)) {
      D[i, j] <- euclidean_sq_na_weighted(X[, i], w, info)
    }
  }
  D
}

#' Impute poorly-observed neuron weight dimensions from SOM grid neighbors
#'
#' @param neurons named list of neurons
#' @param pos_key character position key of neuron to impute
#' @param dim_obs_counts numeric vector of per-dimension observation counts
#' @param all_keys character vector of all neuron keys
#' @param min_obs numeric minimum observations threshold
#' @param max_dist integer maximum grid distance for neighbors
#' @return modified neurons list
impute_neuron_from_neighbors <- function(neurons, pos_key, dim_obs_counts,
                                          all_keys, min_obs = 5.0, max_dist = 2L) {
  pos <- key_to_pos(pos_key)
  input_dim <- length(neurons[[pos_key]]$weights)

  neighbors <- list()
  for (key in all_keys) {
    if (key == pos_key) next
    other_pos <- key_to_pos(key)
    d <- grid_distance_hex(pos, other_pos)
    if (d <= max_dist) {
      neighbors <- c(neighbors, list(list(key = key, dist = d)))
    }
  }

  if (length(neighbors) == 0) return(neurons)

  for (d in seq_len(input_dim)) {
    if (dim_obs_counts[d] < min_obs) {
      numerator <- 0.0
      denominator <- 0.0
      for (nb in neighbors) {
        w <- 1.0 / nb$dist
        numerator <- numerator + w * neurons[[nb$key]]$weights[d]
        denominator <- denominator + w
      }
      if (denominator > .Machine$double.eps) {
        neurons[[pos_key]]$weights[d] <- numerator / denominator
      }
    }
  }
  neurons
}

#' NA-aware batch weight update with neighbor-based imputation
#'
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param bmu_keys character vector of BMU keys per sample
#' @param sigma numeric neighborhood width
#' @param miss_info missingness info list
#' @return modified neurons list
update_weights_batch_na_impute <- function(neurons, X, bmu_keys, sigma, miss_info) {
  n_samples <- ncol(X)
  input_dim <- nrow(X)
  all_keys <- names(neurons)
  obs_counts <- list()

  for (key in all_keys) {
    pos <- key_to_pos(key)
    numerator <- rep(0.0, input_dim)
    denominator <- rep(0.0, input_dim)

    for (i in seq_len(n_samples)) {
      bmu_pos <- key_to_pos(bmu_keys[i])
      d <- grid_distance_hex(pos, bmu_pos)
      h <- gaussian_neighborhood(d, sigma)

      for (dd in seq_len(input_dim)) {
        xval <- X[dd, i]
        if (!is.na(xval)) {
          w <- miss_info$obs_weights[dd]
          numerator[dd] <- numerator[dd] + h * w * xval
          denominator[dd] <- denominator[dd] + h * w
        }
      }
    }

    for (dd in seq_len(input_dim)) {
      if (denominator[dd] > .Machine$double.eps) {
        neurons[[key]]$weights[dd] <- numerator[dd] / denominator[dd]
      }
    }
    obs_counts[[key]] <- denominator
  }

  # Impute poorly-observed dimensions from neighbors
  for (key in all_keys) {
    counts <- obs_counts[[key]]
    max_count <- max(counts)
    relative_threshold <- if (max_count > 0) max_count * 0.1 else 0.0
    threshold <- max(1.0, relative_threshold)
    neurons <- impute_neuron_from_neighbors(neurons, key, counts, all_keys,
                                             min_obs = threshold)
  }

  neurons
}

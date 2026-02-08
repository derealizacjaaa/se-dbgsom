# batch_update.R - Batch weight updates and growth threshold computation
#
# All functions that modify neurons RETURN the modified list.

#' Batch weight update using Gaussian neighborhood (complete data, no NA)
#'
#' For each neuron j: w_j = sum_i h(j, bmu_i) * x_i / sum_i h(j, bmu_i)
#'
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param bmu_keys character vector of BMU position keys per sample
#' @param sigma numeric neighborhood width
#' @return modified neurons list
update_weights_batch_complete <- function(neurons, X, bmu_keys, sigma) {
  n_samples <- ncol(X)
  input_dim <- nrow(X)

  for (key in names(neurons)) {
    pos <- key_to_pos(key)
    numerator <- rep(0.0, input_dim)
    denominator <- 0.0

    for (i in seq_len(n_samples)) {
      bmu_pos <- key_to_pos(bmu_keys[i])
      d <- grid_distance_hex(pos, bmu_pos)
      h <- gaussian_neighborhood(d, sigma)

      numerator <- numerator + h * X[, i]
      denominator <- denominator + h
    }

    if (denominator > .Machine$double.eps) {
      neurons[[key]]$weights <- numerator / denominator
    }
  }
  neurons
}

#' Batch weight update with per-dimension NA handling
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param bmu_keys character vector of BMU position keys
#' @param sigma numeric neighborhood width
#' @return modified neurons list
update_weights_batch_na <- function(neurons, X, bmu_keys, sigma) {
  n_samples <- ncol(X)
  input_dim <- nrow(X)

  for (key in names(neurons)) {
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
          numerator[dd] <- numerator[dd] + h * xval
          denominator[dd] <- denominator[dd] + h
        }
      }
    }

    for (dd in seq_len(input_dim)) {
      if (denominator[dd] > .Machine$double.eps) {
        neurons[[key]]$weights[dd] <- numerator[dd] / denominator[dd]
      }
    }
  }
  neurons
}

#' Auto-dispatch batch weight update based on NA presence
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param bmu_keys character vector of BMU position keys
#' @param sigma numeric neighborhood width
#' @return modified neurons list
update_weights_batch <- function(neurons, X, bmu_keys, sigma) {
  if (has_na(X)) {
    update_weights_batch_na(neurons, X, bmu_keys, sigma)
  } else {
    update_weights_batch_complete(neurons, X, bmu_keys, sigma)
  }
}

# ============================================================================
# Growth Threshold
# ============================================================================

#' Create a spreading factor growth threshold method
#' @param sf numeric spreading factor in (0, 1]
#' @return list with type and sf
spreading_factor <- function(sf = 0.5) {
  stopifnot(sf > 0, sf <= 1)
  list(type = "spreading_factor", sf = sf)
}

#' Create a statistical error growth threshold method
#' @param lambda numeric regulation coefficient (> 0)
#' @return list with type and lambda
statistical_error <- function(lambda = 1.0) {
  stopifnot(lambda > 0)
  list(type = "statistical_error", lambda = lambda)
}

#' Compute growth threshold
#'
#' SpreadingFactor: GT = -log(sf) * d
#' StatisticalError: GT = lambda * sqrt(sum(std_i^2))
#'
#' @param gt_method growth threshold method (list with type)
#' @param input_dim integer data dimensionality
#' @param X matrix (input_dim x n_samples), required for statistical_error
#' @return numeric growth threshold
compute_growth_threshold <- function(gt_method, input_dim, X = NULL) {
  if (gt_method$type == "spreading_factor") {
    return(-log(gt_method$sf) * input_dim)
  }

  if (gt_method$type == "statistical_error") {
    if (is.null(X)) stop("StatisticalError method requires training data X")

    stds <- numeric(input_dim)
    for (d in seq_len(input_dim)) {
      vals <- X[d, ]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 1) {
        stds[d] <- sd(vals)
      }
    }
    return(gt_method$lambda * sqrt(sum(stds^2)))
  }

  stop("Unknown gt_method type: ", gt_method$type)
}

# ============================================================================
# Quantization Error
# ============================================================================

#' Compute mean quantization error
#'
#' QE = (1/N) * sum_i ||x_i - w_bmu_i||
#'
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @return numeric mean QE
compute_quantization_error <- function(neurons, X) {
  res <- find_bmus_with_distances(neurons, X)
  mean(res$distances)
}

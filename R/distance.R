# distance.R - Distance computations for DBGSOM

#' Squared Euclidean distance between two vectors
#' @param x numeric vector
#' @param y numeric vector
#' @return scalar squared distance
euclidean_sq <- function(x, y) {
  d <- x - y
  sum(d * d)
}

#' Euclidean distance between two vectors
#' @param x numeric vector
#' @param y numeric vector
#' @return scalar distance
euclidean_dist <- function(x, y) {
  sqrt(euclidean_sq(x, y))
}

#' Squared Euclidean distance, NA-aware with scaling
#'
#' Only computes over dimensions where both vectors are non-NA.
#' Scales by total_dims / valid_dims to remain comparable.
#' Returns Inf if no valid dimensions.
#'
#' @param x numeric vector
#' @param y numeric vector
#' @return scalar squared distance
euclidean_sq_na <- function(x, y) {
  valid <- !is.na(x) & !is.na(y)
  n_valid <- sum(valid)
  if (n_valid == 0L) return(Inf)
  d <- x[valid] - y[valid]
  total_dim <- length(x)
  sum(d * d) * (total_dim / n_valid)
}

#' NA-aware Euclidean distance
#' @param x numeric vector
#' @param y numeric vector
#' @return scalar distance
euclidean_dist_na <- function(x, y) {
  sqrt(euclidean_sq_na(x, y))
}

#' Pairwise squared distances between data X and weight matrix W
#'
#' Uses the identity: ||x-w||^2 = ||x||^2 + ||w||^2 - 2*x'*w
#'
#' @param X matrix (input_dim x n_samples)
#' @param W matrix (input_dim x n_neurons)
#' @return matrix (n_samples x n_neurons) of squared distances
pairwise_dist_sq <- function(X, W) {
  x_norms <- colSums(X^2)  # length n_samples
  w_norms <- colSums(W^2)  # length n_neurons
  # D = -2 * X' * W, then add norms
  D <- -2.0 * crossprod(X, W)  # n_samples x n_neurons
  D <- sweep(D, 1, x_norms, "+")
  D <- sweep(D, 2, w_norms, "+")
  # Clamp small negatives from floating point
  D[D < 0] <- 0
  D
}

#' Pairwise squared distances, NA-aware (element-wise, no BLAS)
#' @param X matrix (input_dim x n_samples)
#' @param W matrix (input_dim x n_neurons)
#' @return matrix (n_samples x n_neurons)
pairwise_dist_sq_na <- function(X, W) {
  n_samples <- ncol(X)
  n_neurons <- ncol(W)
  D <- matrix(NA_real_, nrow = n_samples, ncol = n_neurons)
  for (j in seq_len(n_neurons)) {
    w <- W[, j]
    for (i in seq_len(n_samples)) {
      D[i, j] <- euclidean_sq_na(X[, i], w)
    }
  }
  D
}

#' Check if matrix has any NA values
#' @param X matrix
#' @return logical
has_na <- function(X) {
  anyNA(X)
}

#' Auto-dispatch pairwise distances: BLAS or NA-aware
#' @param X matrix (input_dim x n_samples)
#' @param W matrix (input_dim x n_neurons)
#' @return matrix (n_samples x n_neurons)
pairwise_dist_sq_auto <- function(X, W) {
  if (has_na(X)) {
    pairwise_dist_sq_na(X, W)
  } else {
    pairwise_dist_sq(X, W)
  }
}

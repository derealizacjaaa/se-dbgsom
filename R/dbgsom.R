# dbgsom.R - Main DBGSOM class: constructor, fit, predict, transform
#
# Directed Batch Growing Self-Organizing Map
# Reference: Vasighi & Amini (2017)
#
# Data convention: X is a matrix with rows = observations, columns = features.
# Internally, data is stored as input_dim x n_samples (transposed) for
# consistency with the algorithm's column-vector convention.

#' Generic fit function
#' @param object model object
#' @param ... additional arguments
#' @export
fit <- function(object, ...) UseMethod("fit")

#' Create a new DBGSOM model
#'
#' @param input_dim integer dimension of input data
#' @param gt_method growth threshold method: spreading_factor() or statistical_error()
#' @param max_neurons integer maximum number of neurons (default 100)
#' @param n_iter integer training iterations (default 100)
#' @param init_size integer vector c(rows, cols) for initial grid (default c(2,2))
#' @return S3 object of class "dbgsom"
#'
#' @examples
#' som <- dbgsom(input_dim = 4)
#' som <- dbgsom(input_dim = 4, gt_method = statistical_error(1.0))
#' som <- dbgsom(input_dim = 10, gt_method = spreading_factor(0.5),
#'               max_neurons = 200, n_iter = 150)
dbgsom <- function(input_dim,
                   gt_method = spreading_factor(0.5),
                   max_neurons = 100L,
                   n_iter = 100L,
                   init_size = c(2L, 2L)) {
  stopifnot(input_dim > 0, max_neurons > 0, n_iter > 0)
  stopifnot(length(init_size) == 2, all(init_size > 0))

  neurons <- init_topology(input_dim, init_size[1], init_size[2])

  obj <- list(
    neurons = neurons,
    input_dim = input_dim,
    gt_method = gt_method,
    max_neurons = max_neurons,
    n_iter = n_iter,
    trained = FALSE,
    missingness_info = NULL
  )
  class(obj) <- "dbgsom"
  obj
}

#' Print method for dbgsom
#' @param x dbgsom object
#' @param ... additional arguments (ignored)
print.dbgsom <- function(x, ...) {
  cat("DBGSOM (Directed Batch Growing Self-Organizing Map)\n")
  cat("  Input dim:   ", x$input_dim, "\n")
  cat("  GT method:   ", x$gt_method$type, "\n")
  if (x$gt_method$type == "spreading_factor") {
    cat("  SF:          ", x$gt_method$sf, "\n")
  } else {
    cat("  Lambda:      ", x$gt_method$lambda, "\n")
  }
  cat("  Max neurons: ", x$max_neurons, "\n")
  cat("  Iterations:  ", x$n_iter, "\n")
  cat("  Neurons:     ", length(x$neurons), "\n")
  cat("  Trained:     ", x$trained, "\n")
  invisible(x)
}

#' Train the DBGSOM on data X
#'
#' Two-phase training:
#' 1. Coarse phase (with growth): large neighborhood, neurons added
#' 2. Fine phase (no growth): small neighborhood, refine positions
#'
#' @param object dbgsom object
#' @param X numeric matrix (n_samples x input_dim) or (input_dim x n_samples)
#' @param ... additional arguments (ignored)
#' @return trained dbgsom object
fit.dbgsom <- function(object, X, ...) {
  # Ensure X is input_dim x n_samples (columns are observations)
  X <- ensure_col_major(X, object$input_dim)
  stopifnot(nrow(X) == object$input_dim)

  neurons <- object$neurons

  # Compute growth threshold
  gt <- compute_growth_threshold(object$gt_method, object$input_dim, X)

  # Compute missingness info for NA-aware training
  miss_info <- if (has_na(X)) compute_missingness_info(X) else NULL

  # Split into coarse and fine phases
  n_coarse <- object$n_iter %/% 2L
  n_fine <- object$n_iter - n_coarse

  # Phase 1: Coarse training with growth
  if (n_coarse > 0) {
    sr <- compute_sigma_range(neurons)
    sigma_start <- sr$sigma_start
    sigma_end <- sr$sigma_end

    for (epoch in seq_len(n_coarse)) {
      sigma <- decay_sigma(epoch, n_coarse, sigma_start, sigma_end)
      bmu_keys <- dispatch_find_bmus(neurons, X, miss_info)

      if (!is.null(miss_info)) {
        neurons <- update_weights_batch_na_impute(neurons, X, bmu_keys, sigma, miss_info)
      } else {
        neurons <- update_weights_batch(neurons, X, bmu_keys, sigma)
      }

      neurons <- reset_errors(neurons)
      neurons <- accumulate_errors(neurons, X, bmu_keys, miss_info)
      neurons <- distribute_errors(neurons, gt)
      grow_result <- grow(neurons, gt, object$max_neurons)
      neurons <- grow_result$neurons
    }
  }

  # Phase 2: Fine training (no growth)
  if (n_fine > 0) {
    sigma_fine_start <- compute_sigma(neurons) * 0.3
    sigma_fine_end <- 0.5

    for (epoch in seq_len(n_fine)) {
      sigma <- decay_sigma(epoch, n_fine, sigma_fine_start, sigma_fine_end)
      bmu_keys <- dispatch_find_bmus(neurons, X, miss_info)

      if (!is.null(miss_info)) {
        neurons <- update_weights_batch_na_impute(neurons, X, bmu_keys, sigma, miss_info)
      } else {
        neurons <- update_weights_batch(neurons, X, bmu_keys, sigma)
      }
    }
  }

  # Cleanup: prune dead neurons
  neurons <- reset_errors(neurons)
  bmu_keys <- dispatch_find_bmus(neurons, X, miss_info)
  neurons <- accumulate_errors(neurons, X, bmu_keys, miss_info)
  prune_result <- prune_dead_neurons(neurons)
  neurons <- prune_result$neurons

  object$neurons <- neurons
  object$missingness_info <- miss_info
  object$trained <- TRUE
  object
}

#' Predict BMU positions for new data
#'
#' @param object trained dbgsom object
#' @param newdata numeric matrix (n_samples x input_dim)
#' @param ... additional arguments (ignored)
#' @return character vector of BMU position keys ("x,y")
predict.dbgsom <- function(object, newdata, ...) {
  stopifnot(object$trained)
  X <- ensure_col_major(newdata, object$input_dim)
  dispatch_find_bmus(object$neurons, X, object$missingness_info)
}

#' Transform data to BMU coordinates
#'
#' @param object trained dbgsom object
#' @param newdata numeric matrix (n_samples x input_dim)
#' @param ... additional arguments (ignored)
#' @return matrix (n_samples x 2) with columns x, y
transform.dbgsom <- function(object, newdata, ...) {
  stopifnot(object$trained)
  bmu_keys <- predict.dbgsom(object, newdata)
  coords <- t(vapply(bmu_keys, key_to_pos, integer(2)))
  colnames(coords) <- c("x", "y")
  coords
}

#' Predict BMU positions with distances
#'
#' @param som trained dbgsom object
#' @param X numeric matrix (n_samples x input_dim)
#' @return list with bmu_keys and distances
predict_with_distances <- function(som, X) {
  stopifnot(som$trained)
  X <- ensure_col_major(X, som$input_dim)
  find_bmus_with_distances(som$neurons, X)
}

#' Get neuron positions and weight matrix
#'
#' @param som dbgsom object
#' @return list with positions (list of int vectors) and weights (matrix: n_neurons x input_dim)
get_neuron_weights <- function(som) {
  keys <- names(som$neurons)
  positions <- lapply(keys, key_to_pos)
  n <- length(keys)
  if (n == 0) return(list(positions = list(), weights = matrix(nrow = 0, ncol = 0)))
  W <- matrix(NA_real_, nrow = n, ncol = som$input_dim)
  for (i in seq_along(keys)) {
    W[i, ] <- som$neurons[[keys[i]]]$weights
  }
  list(positions = positions, weights = W)
}

# ============================================================================
# Internal Helpers
# ============================================================================

#' Ensure data is in column-major format (input_dim x n_samples)
#'
#' Accepts n_samples x input_dim (rows=obs) and transposes if needed.
#'
#' @param X numeric matrix
#' @param input_dim expected number of features
#' @return matrix (input_dim x n_samples)
ensure_col_major <- function(X, input_dim) {
  X <- as.matrix(X)
  if (nrow(X) == input_dim) {
    return(X)
  }
  if (ncol(X) == input_dim) {
    return(t(X))
  }
  stop("Data dimensions (", nrow(X), " x ", ncol(X),
       ") do not match input_dim (", input_dim, ")")
}

# growth.R - Growth, error distribution, and pruning for DBGSOM
#
# All functions that modify neurons RETURN the modified list.
# R lists have copy-on-modify semantics, so callers must reassign.

# ============================================================================
# Error Accumulation
# ============================================================================

#' Reset errors and hit counts for all neurons
#' @param neurons named list of neurons
#' @return modified neurons list
reset_errors <- function(neurons) {
  for (key in names(neurons)) {
    neurons[[key]]$error <- 0.0
    neurons[[key]]$hits <- 0L
  }
  neurons
}

#' Accumulate quantization errors for each neuron based on BMU assignments
#' @param neurons named list of neurons
#' @param X matrix (input_dim x n_samples)
#' @param bmu_keys character vector of BMU position keys
#' @param miss_info missingness info list or NULL
#' @return modified neurons list
accumulate_errors <- function(neurons, X, bmu_keys, miss_info = NULL) {
  n_samples <- ncol(X)

  for (i in seq_len(n_samples)) {
    key <- bmu_keys[i]
    w <- neurons[[key]]$weights
    x <- X[, i]

    if (!is.null(miss_info) && any(is.na(x))) {
      dist_sq <- euclidean_sq_na_weighted(w, x, miss_info)
    } else if (any(is.na(x))) {
      dist_sq <- euclidean_sq_na(w, x)
    } else {
      dist_sq <- euclidean_sq(w, x)
    }

    d <- sqrt(dist_sq)
    if (is.finite(d)) {
      neurons[[key]]$error <- neurons[[key]]$error + d
      neurons[[key]]$hits <- neurons[[key]]$hits + 1L
    }
  }
  neurons
}

# ============================================================================
# Error Distribution
# ============================================================================

#' Distribute errors from non-boundary to boundary neurons
#'
#' For non-boundary neuron with E >= GT:
#' - Each boundary neighbor gets error/12
#' - The non-boundary neuron retains error/2
#'
#' @param neurons named list of neurons
#' @param gt numeric growth threshold
#' @return modified neurons list
distribute_errors <- function(neurons, gt) {
  keys <- names(neurons)
  for (key in keys) {
    err <- neurons[[key]]$error
    pos <- key_to_pos(key)

    if (err >= gt && !is_boundary(neurons, pos)) {
      nbrs <- get_neighbors_hex(pos)
      boundary_nbrs <- character(0)
      for (nbr in nbrs) {
        nbr_key <- pos_to_key(nbr)
        if (!is.null(neurons[[nbr_key]]) && is_boundary(neurons, nbr)) {
          boundary_nbrs <- c(boundary_nbrs, nbr_key)
        }
      }

      if (length(boundary_nbrs) > 0) {
        error_share <- err / 12.0
        for (nbr_key in boundary_nbrs) {
          neurons[[nbr_key]]$error <- neurons[[nbr_key]]$error + error_share
        }
        neurons[[key]]$error <- err / 2.0
      }
    }
  }
  neurons
}

# ============================================================================
# Growth Candidates
# ============================================================================

#' Find boundary neurons with error exceeding growth threshold, sorted by error
#' @param neurons named list of neurons
#' @param gt numeric growth threshold
#' @return character vector of position keys (highest error first)
find_growth_candidates <- function(neurons, gt) {
  candidates <- character(0)
  errors <- numeric(0)

  for (key in names(neurons)) {
    pos <- key_to_pos(key)
    if (neurons[[key]]$error > gt && is_boundary(neurons, pos)) {
      candidates <- c(candidates, key)
      errors <- c(errors, neurons[[key]]$error)
    }
  }

  if (length(candidates) == 0) return(character(0))
  candidates[order(errors, decreasing = TRUE)]
}

# ============================================================================
# Directed Growth (Vasighi & Amini 2017)
# ============================================================================

#' Select the best growth direction from a boundary neuron
#'
#' Considers 1st and 2nd-order neighbors to bias growth toward high-error regions.
#'
#' @param neurons named list of neurons
#' @param boundary_key character position key of boundary neuron
#' @return list(pos_key, weights) or NULL if no growth possible
select_growth_direction <- function(neurons, boundary_key) {
  pos <- key_to_pos(boundary_key)
  parent <- neurons[[boundary_key]]
  allowed <- get_empty_neighbors(neurons, pos)

  if (length(allowed) == 0) return(NULL)

  # 1st-order occupied neighbors
  nbr1 <- get_neighbors_at_distance(neurons, pos, 1L)
  # 2nd-order occupied neighbors
  nbr2 <- get_neighbors_at_distance(neurons, pos, 2L)

  # Sort nbr1 by error (highest first)
  if (length(nbr1) > 1) {
    errs <- vapply(nbr1, function(p) neurons[[pos_to_key(p)]]$error, numeric(1))
    nbr1 <- nbr1[order(errs, decreasing = TRUE)]
  }
  if (length(nbr2) > 1) {
    errs <- vapply(nbr2, function(p) neurons[[pos_to_key(p)]]$error, numeric(1))
    nbr2 <- nbr2[order(errs, decreasing = TRUE)]
  }

  n_nbr1 <- length(nbr1)

  if (n_nbr1 == 0) {
    new_pos <- allowed[[1]]
    weights <- parent$weights + rnorm(length(parent$weights)) * 0.01
    return(list(pos_key = pos_to_key(new_pos), weights = weights))
  }

  if (n_nbr1 <= 2) {
    return(grow_few_neighbors(neurons, pos, parent, allowed, nbr1, nbr2))
  }

  if (n_nbr1 <= 4) {
    return(grow_medium_neighbors(neurons, pos, parent, allowed, nbr1))
  }

  return(grow_many_neighbors(neurons, pos, parent, allowed, nbr1))
}

#' Growth when 1-2 direct neighbors
grow_few_neighbors <- function(neurons, pos, parent, allowed, nbr1, nbr2) {
  sel_nbr2 <- list()
  for (p in nbr2) {
    for (a in allowed) {
      if (grid_distance_hex(p, a) == 1L) {
        sel_nbr2 <- c(sel_nbr2, list(p))
        break
      }
    }
  }

  if (length(sel_nbr2) > 0) {
    if (length(sel_nbr2) > 1) {
      errs <- vapply(sel_nbr2, function(p) neurons[[pos_to_key(p)]]$error, numeric(1))
      sel_nbr2 <- sel_nbr2[order(errs, decreasing = TRUE)]
    }
    best_nbr2_pos <- sel_nbr2[[1]]
    best_nbr2 <- neurons[[pos_to_key(best_nbr2_pos)]]
    dists <- vapply(allowed, function(a) grid_distance_hex(a, best_nbr2_pos), integer(1))
    new_pos <- allowed[[which.min(dists)]]
    weights <- (parent$weights + best_nbr2$weights) / 2.0
  } else {
    nbr1_pos <- nbr1[[1]]
    nbr <- neurons[[pos_to_key(nbr1_pos)]]
    dists <- vapply(allowed, function(a) grid_distance_hex(a, nbr1_pos), integer(1))
    new_pos <- allowed[[which.max(dists)]]
    weights <- 2.0 * parent$weights - nbr$weights
  }

  list(pos_key = pos_to_key(new_pos), weights = weights)
}

#' Growth when 3-4 direct neighbors
grow_medium_neighbors <- function(neurons, pos, parent, allowed, nbr1) {
  max_err_nbr_pos <- nbr1[[1]]
  min_err_nbr <- neurons[[pos_to_key(nbr1[[length(nbr1)]])]]

  dists <- vapply(allowed, function(a) grid_distance_hex(a, max_err_nbr_pos), integer(1))
  new_pos <- allowed[[which.min(dists)]]
  weights <- 2.0 * parent$weights - min_err_nbr$weights

  list(pos_key = pos_to_key(new_pos), weights = weights)
}

#' Growth when 5 direct neighbors
grow_many_neighbors <- function(neurons, pos, parent, allowed, nbr1) {
  new_pos <- allowed[[1]]

  opposite <- list()
  for (n in nbr1) {
    if (grid_distance_hex(n, new_pos) == 2L) {
      opposite <- c(opposite, list(n))
    }
  }

  if (length(opposite) > 0) {
    opp_nbr <- neurons[[pos_to_key(opposite[[1]])]]
    weights <- 2.0 * parent$weights - opp_nbr$weights
  } else {
    avg_w <- Reduce("+", lapply(nbr1, function(n) neurons[[pos_to_key(n)]]$weights)) / length(nbr1)
    weights <- 2.0 * parent$weights - avg_w
  }

  list(pos_key = pos_to_key(new_pos), weights = weights)
}

# ============================================================================
# Growth Step
# ============================================================================

#' Grow neurons from high-error boundary neurons using directed growth
#'
#' @param neurons named list of neurons
#' @param gt numeric growth threshold
#' @param max_neurons integer maximum neurons allowed
#' @return list with neurons (modified) and n_added (integer)
grow <- function(neurons, gt, max_neurons) {
  candidates <- find_growth_candidates(neurons, gt)
  n_added <- 0L

  for (key in candidates) {
    if (length(neurons) >= max_neurons) break

    result <- select_growth_direction(neurons, key)
    if (!is.null(result)) {
      neurons[[result$pos_key]] <- new_neuron(result$weights)
      neurons[[key]]$error <- 0.0
      n_added <- n_added + 1L
    }
  }

  list(neurons = neurons, n_added = n_added)
}

# ============================================================================
# Pruning
# ============================================================================

#' Find dead neurons (hit count < min_hits)
#' @param neurons named list of neurons
#' @param min_hits integer minimum hit threshold
#' @return character vector of dead neuron keys
find_dead_neurons <- function(neurons, min_hits = 1L) {
  dead <- character(0)
  for (key in names(neurons)) {
    if (neurons[[key]]$hits < min_hits) {
      dead <- c(dead, key)
    }
  }
  dead
}

#' Remove dead neurons, keeping at least min_neurons
#' @param neurons named list of neurons
#' @param min_hits integer minimum hit threshold
#' @param min_neurons integer minimum neurons to keep
#' @return list with neurons (modified) and n_removed (integer)
prune_dead_neurons <- function(neurons, min_hits = 1L, min_neurons = 4L) {
  dead <- find_dead_neurons(neurons, min_hits)
  n_removed <- 0L

  for (key in dead) {
    if (length(neurons) > min_neurons) {
      neurons[[key]] <- NULL
      n_removed <- n_removed + 1L
    }
  }

  list(neurons = neurons, n_removed = n_removed)
}

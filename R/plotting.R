# plotting.R - Visualization data extraction for DBGSOM
#
# Computes U-matrix, component planes, hit counts, and topology edges.
# Returns data structures suitable for plotting with base R or ggplot2.

# ============================================================================
# U-Matrix
# ============================================================================

#' Compute the U-matrix (unified distance matrix)
#'
#' Average distance between each neuron and its occupied hex neighbors.
#' High values = cluster boundaries, low values = cluster interiors.
#'
#' @param som dbgsom object (trained)
#' @return named numeric vector keyed by position "x,y"
compute_umatrix <- function(som) {
  neurons <- som$neurons
  umat <- numeric(0)

  for (key in names(neurons)) {
    pos <- key_to_pos(key)
    nbrs <- get_neighbors_hex(pos)
    dists <- numeric(0)

    for (nbr in nbrs) {
      nbr_key <- pos_to_key(nbr)
      if (!is.null(neurons[[nbr_key]])) {
        d <- euclidean_dist(neurons[[key]]$weights, neurons[[nbr_key]]$weights)
        dists <- c(dists, d)
      }
    }

    umat[key] <- if (length(dists) == 0) 0.0 else mean(dists)
  }

  umat
}

#' Compute expanded U-matrix with per-edge distances
#'
#' @param som dbgsom object (trained)
#' @param log_transform logical, apply log transform to distances
#' @return list with neuron_umat (named vector) and edge_umat (named list)
compute_expanded_umatrix <- function(som, log_transform = FALSE) {
  neuron_umat <- compute_umatrix(som)
  neurons <- som$neurons
  edge_umat <- list()

  for (key in names(neurons)) {
    pos <- key_to_pos(key)
    for (nbr in get_neighbors_hex(pos)) {
      nbr_key <- pos_to_key(nbr)
      if (!is.null(neurons[[nbr_key]])) {
        # Canonical edge key (sorted)
        edge_key <- if (key < nbr_key) paste0(key, "|", nbr_key) else paste0(nbr_key, "|", key)
        if (is.null(edge_umat[[edge_key]])) {
          d <- euclidean_dist(neurons[[key]]$weights, neurons[[nbr_key]]$weights)
          edge_umat[[edge_key]] <- if (log_transform) log(d + .Machine$double.eps) else d
        }
      }
    }
  }

  if (log_transform) {
    neuron_umat <- log(neuron_umat + .Machine$double.eps)
  }

  list(neuron_umat = neuron_umat, edge_umat = edge_umat)
}

#' Compute U* matrix: U-matrix combined with density (hit counts)
#'
#' U*(i) = U(i) * (1 - normalized_density(i))
#'
#' @param som dbgsom object (trained)
#' @param X matrix (n_samples x input_dim)
#' @return named numeric vector
compute_ustar_matrix <- function(som, X) {
  umat <- compute_umatrix(som)
  hits <- compute_hit_counts(som, X)

  if (length(umat) == 0) return(numeric(0))

  max_hits <- max(hits)
  if (max_hits == 0) return(umat)

  ustar <- umat
  for (key in names(umat)) {
    norm_density <- hits[key] / max_hits
    ustar[key] <- umat[key] * (1.0 - norm_density)
  }
  ustar
}

# ============================================================================
# Component Planes
# ============================================================================

#' Extract component planes for all input dimensions
#'
#' @param som dbgsom object (trained)
#' @return list of named numeric vectors (one per dimension)
compute_component_planes <- function(som) {
  neurons <- som$neurons
  input_dim <- som$input_dim
  planes <- vector("list", input_dim)

  for (d in seq_len(input_dim)) {
    plane <- numeric(0)
    for (key in names(neurons)) {
      plane[key] <- neurons[[key]]$weights[d]
    }
    planes[[d]] <- plane
  }

  planes
}

# ============================================================================
# Hit Counts
# ============================================================================

#' Count how many samples map to each neuron
#'
#' @param som dbgsom object (trained)
#' @param X matrix (n_samples x input_dim)
#' @return named integer vector keyed by position "x,y"
compute_hit_counts <- function(som, X) {
  X <- ensure_col_major(X, som$input_dim)
  bmu_keys <- dispatch_find_bmus(som$neurons, X, som$missingness_info)

  counts <- setNames(rep(0L, length(som$neurons)), names(som$neurons))
  for (key in bmu_keys) {
    counts[key] <- counts[key] + 1L
  }
  counts
}

# ============================================================================
# Topology Edges
# ============================================================================

#' Get all edges in the topology graph
#'
#' @param som dbgsom object
#' @return list of edge pairs, each c("x1,y1", "x2,y2")
compute_topology_edges <- function(som) {
  neurons <- som$neurons
  edges <- list()
  seen <- character(0)

  for (key in names(neurons)) {
    pos <- key_to_pos(key)
    for (nbr in get_neighbors_hex(pos)) {
      nbr_key <- pos_to_key(nbr)
      if (!is.null(neurons[[nbr_key]])) {
        edge_id <- if (key < nbr_key) paste0(key, "|", nbr_key) else paste0(nbr_key, "|", key)
        if (!(edge_id %in% seen)) {
          seen <- c(seen, edge_id)
          edges <- c(edges, list(c(key, nbr_key)))
        }
      }
    }
  }

  edges
}

# ============================================================================
# Combined Plotting Data
# ============================================================================

#' Extract all visualization data from a trained DBGSOM
#'
#' @param som trained dbgsom object
#' @param X matrix (n_samples x input_dim) for hit counts
#' @return list with positions, weights, umatrix, hit_counts, edges, bounds
get_plotting_data <- function(som, X) {
  nw <- get_neuron_weights(som)
  list(
    positions = nw$positions,
    weights = nw$weights,
    umatrix = compute_umatrix(som),
    hit_counts = compute_hit_counts(som, X),
    edges = compute_topology_edges(som),
    bounds = grid_bounds(som$neurons)
  )
}

# ============================================================================
# ASCII Visualization
# ============================================================================

#' Generate ASCII representation of the SOM grid
#'
#' @param som dbgsom object
#' @param show_hits logical, show hit counts instead of presence
#' @return character string
format_grid_ascii <- function(som, show_hits = FALSE) {
  b <- grid_bounds(som$neurons)
  lines <- character(0)
  lines <- c(lines, sprintf("Grid (%dx%d):",
                             b$x_max - b$x_min + 1, b$y_max - b$y_min + 1))

  header <- "  "
  for (x in b$x_min:b$x_max) header <- paste0(header, sprintf("%2d", x))
  lines <- c(lines, header)

  for (y in b$y_min:b$y_max) {
    row <- sprintf("%2d", y)
    for (x in b$x_min:b$x_max) {
      key <- pos_key(x, y)
      if (!is.null(som$neurons[[key]])) {
        if (show_hits) {
          h <- som$neurons[[key]]$hits
          row <- paste0(row, sprintf("%2s", if (h > 9) "+" else as.character(h)))
        } else {
          row <- paste0(row, " *")
        }
      } else {
        row <- paste0(row, " .")
      }
    }
    lines <- c(lines, row)
  }

  paste(lines, collapse = "\n")
}

#' Generate a text summary of the trained SOM
#'
#' @param som dbgsom object
#' @param X matrix for QE computation (optional)
#' @return character string
format_summary <- function(som, X = NULL) {
  lines <- character(0)
  lines <- c(lines, "DBGSOM Summary")
  lines <- c(lines, paste(rep("=", 40), collapse = ""))
  lines <- c(lines, "")
  lines <- c(lines, "Configuration:")
  lines <- c(lines, sprintf("  Input dimension: %d", som$input_dim))
  if (som$gt_method$type == "spreading_factor") {
    lines <- c(lines, sprintf("  Spreading factor: %.3f", som$gt_method$sf))
  } else {
    lines <- c(lines, sprintf("  Lambda: %.3f", som$gt_method$lambda))
  }
  lines <- c(lines, sprintf("  Max neurons: %d", som$max_neurons))
  lines <- c(lines, sprintf("  Iterations: %d", som$n_iter))
  lines <- c(lines, "")
  lines <- c(lines, "Topology:")
  lines <- c(lines, sprintf("  Neurons: %d", length(som$neurons)))
  b <- grid_bounds(som$neurons)
  lines <- c(lines, sprintf("  Bounds: x=[%d,%d] y=[%d,%d]",
                             b$x_min, b$x_max, b$y_min, b$y_max))
  lines <- c(lines, "")

  if (som$trained && !is.null(X)) {
    Xc <- ensure_col_major(X, som$input_dim)
    qe <- compute_quantization_error(som$neurons, Xc)
    umat <- compute_umatrix(som)
    hits <- compute_hit_counts(som, X)

    lines <- c(lines, "Quality Metrics:")
    lines <- c(lines, sprintf("  Quantization error: %.4f", qe))
    lines <- c(lines, sprintf("  Mean U-matrix: %.4f", mean(umat)))
    lines <- c(lines, sprintf("  Active neurons: %d / %d",
                               sum(hits > 0), length(som$neurons)))
  } else if (!som$trained) {
    lines <- c(lines, "Status: Not trained")
  }

  paste(lines, collapse = "\n")
}
